import argparse
import copy
import csv
import time
import warnings

import cv2
import numpy as np
import torch
import torch.ao.nn.intrinsic.qat as nniqat
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import (FusedMovingAvgObsFakeQuantize, MovingAverageMinMaxObserver, QConfig, QConfigMapping,
                                   default_fused_per_channel_wt_fake_quant, disable_observer, enable_observer)
from torch.ao.quantization.quantize_fx import convert_fx, prepare_qat_fx
from torchvision.models.resnet import BasicBlock

from stage2.compare_yolo_unet import NAMES, ROOT
from deployment.int8_quant import calib_paths, read_split, unet_input
from stage2.kfold_patient import UNET_TAG
from stage2.kfold_pano import PROJECT
from stage2.kfold_unet_variants import CONFIGS, augment, build_model, fold_paths, to_tensors, val_dice
from stage2.kfold_yolo_unet import UNET_ARGS, dice_loss, load_samples
from stage2.predict_unet import DEVICE

warnings.filterwarnings("ignore")
# R34/512 的 QAT 與定點驗證(2026-10-31,伺服器 GPU + CPU,使用者要求)。NNCF 的 torch 後端在這台機器編不出 GPU 擴充,改用 torch.ao FX:
#   權重 s8 逐通道對稱、活化值 u8 全範圍(與 unet_int8.xml 的 u8 × s8 相同),BN 融進卷積,不再有輸入正規化:
#   灰階 0~1 複製三通道再正規化,改成第一層 1 通道卷積 + 偏置;只有邊界補值不同,forward 全程 ×2 雙線性(check_same 確認預測幾乎不變)。
#   tq_ptq: 插入 FakeQuantize、各折訓練集 --calib 張校正後直接轉換;tq_qat: 再微調 --epochs 個 epoch(BN 統計固定、第 5 epoch 後固定量化範圍,
#   驗證集 Dice 最佳的 epoch,不碰測試集);tq_qat_fx: tq_qat 的整數模型改用定點算術模擬(--mult-bits 位元乘數、--acc-bits 位元累加器飽和)。
#   遮罩存到 compare_results/int8_pano/masks/{cv,test}/<變體>[_s<種子>]/fold<k>/;tq_qat_fx 只跑測試集。
OUT = ROOT / "compare_results" / "int8_pano"
PANO = ROOT / "compare_results" / PROJECT
WEIGHTS = PANO / "weights"
SIZE = CONFIGS[UNET_TAG]["size"]
QCONFIG = QConfig(activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255,
                                                                     dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=False),
                  weight=default_fused_per_channel_wt_fake_quant)


class QBlock(BasicBlock):
    def forward(self, x):
        out = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        identity = x if self.downsample is None else self.downsample(x)
        return self.relu2(out + identity)


class QNet(nn.Module):
    def __init__(self, net):
        super().__init__()
        conv = net.stem[0]
        first = nn.Conv2d(1, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=True)
        scale = conv.weight.data / net.std.view(1, 3, 1, 1)
        first.weight.data = scale.sum(1, keepdim=True)
        first.bias.data = -(scale * net.mean.view(1, 3, 1, 1)).sum((1, 2, 3))
        self.conv1, self.bn1, self.relu1 = first, net.stem[1], nn.ReLU()
        self.pool, self.layers, self.decoders, self.head = net.pool, net.layers, net.decoders, net.head
        for block in self.modules():
            if isinstance(block, BasicBlock):
                block.__class__ = QBlock
                block.relu, block.relu2 = nn.ReLU(), nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        skips = [x]
        x = self.pool(x)
        for layer in self.layers:
            x = layer(x)
            skips.append(x)
        x = skips[4]
        for decoder, skip in zip(self.decoders, (skips[3], skips[2], skips[1], skips[0])):
            x = decoder(torch.cat([F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False), skip], dim=1))
        return self.head(F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False))


def load_float(seed, fold):
    net = build_model({**CONFIGS[UNET_TAG], "pretrained": False})
    suffix = f"_s{seed}" if seed else ""
    net.load_state_dict(torch.load(WEIGHTS / f"{UNET_TAG}{suffix}_fold{fold}.pth", map_location="cpu")["model_state_dict"])
    return net.eval()


def check_same(seed, fold):
    # 第一層邊界的 3 px 補值不同(原本補正規化後的 0,現在補原圖的 0),所以比真實影像的預測,而不是逐值相等
    net = load_float(seed, fold)
    qnet = QNet(copy.deepcopy(net)).eval()
    changed = []
    for path in read_split("test")[:8]:
        x = torch.from_numpy(unet_input(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), SIZE))
        with torch.inference_mode():
            changed.append(float((net(x).argmax(1) != qnet(x).argmax(1)).float().mean()))
    print(f"[s{seed} fold{fold}] 第一層改寫後預測改變的像素 {np.mean(changed):.5f}", flush=True)
    assert np.mean(changed) < 0.005


class FxConv(nn.Module):
    # 定點卷積: 累加器 acc_bits 飽和,乘數 mult_bits 位元,四捨五入右移;double 可精確表示到 2^53
    def __init__(self, q, acc_bits, mult_bits):
        super().__init__()
        self.q, self.acc, self.bits = q, 2.0 ** (acc_bits - 1), mult_bits
        self.relu = "ReLU" in type(q).__name__

    def forward(self, x):
        q = self.q
        w = q.weight()
        ws = w.q_per_channel_scales().double().view(-1, 1, 1, 1)
        wi = w.int_repr().double()
        si, zi = x.q_scale(), x.q_zero_point()
        xi = x.int_repr().double() - zi
        acc = F.conv2d(xi, wi, None, q.stride, q.padding, q.dilation, q.groups)
        if q.bias() is not None:
            acc = acc + torch.round(q.bias().double().view(1, -1, 1, 1) / (si * ws.view(1, -1, 1, 1)))
        acc = acc.clamp(-self.acc, self.acc - 1)
        y = requant(acc, si * ws.view(1, -1, 1, 1) / q.scale, self.bits) + q.zero_point
        low = q.zero_point if self.relu else 0
        return torch._make_per_tensor_quantized_tensor(y.clamp(low, 255).to(torch.uint8), q.scale, q.zero_point)


def requant(acc, multiplier, bits):
    # M = m / 2^(shift),m 為 bits 位元整數
    multiplier = torch.as_tensor(multiplier, dtype=torch.float64)
    exponent = torch.floor(torch.log2(multiplier)) + 1
    m = torch.round(multiplier * 2.0 ** (bits - exponent))
    shift = bits - exponent
    return torch.floor(acc * m / 2.0 ** shift + 0.5)


def fx_add(mult_bits, relu):
    def add(a, b, scale, zero_point):
        y = requant(a.int_repr().double() - a.q_zero_point(), a.q_scale() / scale, mult_bits) \
            + requant(b.int_repr().double() - b.q_zero_point(), b.q_scale() / scale, mult_bits) + zero_point
        return torch._make_per_tensor_quantized_tensor(y.clamp(zero_point if relu else 0, 255).to(torch.uint8), scale, zero_point)
    return add


def to_fixed(model, acc_bits, mult_bits):
    model = copy.deepcopy(model)
    names = 0
    for name, module in list(model.named_modules()):
        if "Conv2d" in type(module).__name__ or "ConvReLU2d" in type(module).__name__:
            parent = model.get_submodule(name.rpartition(".")[0]) if "." in name else model
            setattr(parent, name.rpartition(".")[2], FxConv(module, acc_bits, mult_bits))
            names += 1
    adds = 0
    for node in model.graph.nodes:
        if node.op == "call_function" and getattr(node.target, "__name__", "") in ("add", "add_relu") and "quantized" in str(node.target):
            node.target = fx_add(mult_bits, node.target.__name__ == "add_relu")
            adds += 1
    model.recompile()
    print(f"定點替換: 卷積 {names}、加法 {adds}", flush=True)
    return model


def save_masks(model, variant, fold, sets):
    for dataset, paths in sets.items():
        folder = OUT / "masks" / dataset / variant / f"fold{fold}"
        folder.mkdir(parents=True, exist_ok=True)
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            h, w = image.shape
            with torch.inference_mode():
                logits = model(torch.from_numpy(unet_input(image, SIZE)))
            pred = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False).argmax(1)[0].numpy()
            code = np.zeros((h, w), dtype=np.uint8)
            for c in NAMES:
                code[pred == c + 1] = 1 << c
            cv2.imwrite(str(folder / f"{path.stem}.png"), code)


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, (nniqat.ConvBnReLU2d, nniqat.ConvBn2d)):
            m.freeze_bn_stats()


def convert(state, prepared):
    model = copy.deepcopy(prepared).cpu()
    model.load_state_dict(state)
    return convert_fx(model.eval())


def one(seed, fold, args):
    suffix = f"_s{seed}" if seed else ""
    done = OUT / "models" / f"tq{suffix}_fold{fold}.done"
    if done.exists():
        return
    check_same(seed, fold)
    start = time.time()
    torch.manual_seed(seed)
    prepared = prepare_qat_fx(QNet(load_float(seed, fold)).train(), QConfigMapping().set_global(QCONFIG), (torch.zeros(1, 1, *SIZE),))
    prepared.to(DEVICE)
    freeze_bn(prepared)
    with torch.no_grad():
        for path in calib_paths(args.calib, fold):
            prepared(torch.from_numpy(unet_input(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), SIZE)).to(DEVICE))
    prepared.apply(disable_observer)
    ptq_state = {k: v.detach().cpu().clone() for k, v in prepared.state_dict().items()}

    train_x, train_y = to_tensors(load_samples(fold_paths(fold, "train", PANO)), SIZE)
    val_samples = load_samples(fold_paths(fold, "val", PANO))
    optimizer = torch.optim.AdamW(prepared.parameters(), lr=args.lr, weight_decay=UNET_ARGS["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    best_dice, best_state = val_dice(prepared, val_samples, SIZE), ptq_state
    log = [(0, best_dice)]
    for epoch in range(1, args.epochs + 1):
        prepared.train()
        freeze_bn(prepared)
        if epoch <= 5:
            prepared.apply(enable_observer)
        order = torch.randperm(len(train_x))
        for i in range(0, len(order), UNET_ARGS["batch"]):
            idx = order[i : i + UNET_ARGS["batch"]]
            if len(idx) < 2:
                continue
            x, y = augment(train_x[idx].to(DEVICE), train_y[idx].to(DEVICE), True)
            logits = prepared(x)
            loss = F.cross_entropy(logits, y) + dice_loss(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
        prepared.apply(disable_observer)
        dice = val_dice(prepared, val_samples, SIZE)
        log.append((epoch, dice))
        if dice > best_dice:
            best_dice, best_state = dice, {k: v.detach().cpu().clone() for k, v in prepared.state_dict().items()}
    print(f"[s{seed} fold{fold}] val dice PTQ {log[0][1]:.4f} → 最佳 {best_dice:.4f};訓練 {time.time() - start:.0f} s", flush=True)

    sets = {"cv": read_split(f"fold{fold}_val")[:args.limit], "test": read_split("test")[:args.limit]}
    if seed == 0:  # 留種子 0 的 QAT 狀態,之後改位元寬度用
        torch.save(best_state, OUT / "models" / f"tq_qat_fold{fold}.pth")
    qat = convert(best_state, prepared)
    save_masks(convert(ptq_state, prepared), f"tq_ptq{suffix}", fold, sets)
    save_masks(qat, f"tq_qat{suffix}", fold, sets)
    fixed = to_fixed(qat, args.acc_bits, args.mult_bits)
    save_masks(fixed, f"tq_qat_fx{suffix}", fold, {"test": sets["test"]})
    with open(OUT / "qat_val_log.csv", "a", newline="") as f:
        csv.writer(f).writerows((seed, fold, e, round(d, 5)) for e, d in log)
    done.touch()
    print(f"[s{seed} fold{fold}] 完成,累計 {time.time() - start:.0f} s", flush=True)
    del prepared, qat, fixed
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--calib", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--acc-bits", type=int, default=32)
    parser.add_argument("--mult-bits", type=int, default=16)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.backends.quantized.engine = "fbgemm"
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for fold in args.folds:
            one(seed, fold, args)


if __name__ == "__main__":
    main()

