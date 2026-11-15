import argparse
import copy
import csv
import time
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.ao.quantization import (FusedMovingAvgObsFakeQuantize, MovingAverageMinMaxObserver,
                                   MovingAveragePerChannelMinMaxObserver, QConfig, QConfigMapping,
                                   default_fused_per_channel_wt_fake_quant, disable_observer, enable_observer)
from torch.ao.quantization.quantize_fx import prepare_qat_fx

from stage2.compare_yolo_unet import NAMES
from deployment.int8_quant import calib_paths, read_split, unet_input
from stage2.kfold_unet_variants import augment, fold_paths, to_tensors, val_dice
from stage2.kfold_yolo_unet import UNET_ARGS, dice_loss, load_samples
from stage2.predict_unet import DEVICE
import hardware.qat_fx as qat_fx
from hardware.qat_fx import OUT, PANO, SIZE, QNet, check_same, freeze_bn, load_float

warnings.filterwarnings("ignore")
# 4-bit QAT(2026-11-10,伺服器 GPU,使用者要求;見 EXPERIMENTS.md「硬體: 結構化稀疏與 4-bit 的評估與規劃」)。
#   與 qat_fx.py 相同的 QNet、校正(--calib 張)、QAT 超參數(15 epoch、AdamW 3e-5、BN 統計固定、第 5 epoch 後固定量化範圍、
#   驗證 Dice 最佳的 epoch),只改位元寬度: 權重對稱逐通道 −(2^(b−1)−1)..2^(b−1)−1(4 位元 = −7..7),活化值 0..2^b−1(4 位元 = 0..15)。
#   FBGEMM 沒有 4 位元整數 kernel,所以用 fake-quant 模型(量化再反量化)在 GPU 上評估,不轉成整數模型。
#   對照 tq_qat_fq: 把 qat_fx.py 存下的種子 0 INT8 QAT 狀態(tq_qat_fold<k>.pth)走同一條 fake-quant 路徑,應與 tq_qat 幾乎相同。
#   變體: tq_w<W>a<A>_ptq(只校正)、tq_w<W>a<A>_qat;遮罩存到 compare_results/int8_pano/masks/{cv,test}/<變體>/fold<k>/(種子 0 不加後綴)。


def qconfig(w_bits, a_bits):
    # 8 位元權重沿用 qat_fx.py 的 default_fused_per_channel_wt_fake_quant(−128..127),對照組才會完全相同
    wq = 2 ** (w_bits - 1) - 1
    weight = default_fused_per_channel_wt_fake_quant if w_bits == 8 else FusedMovingAvgObsFakeQuantize.with_args(
        observer=MovingAveragePerChannelMinMaxObserver, quant_min=-wq, quant_max=wq, dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric, ch_axis=0, reduce_range=False)
    return QConfig(
        activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0,
                                                           quant_max=2 ** a_bits - 1, dtype=torch.quint8,
                                                           qscheme=torch.per_tensor_affine, reduce_range=False),
        weight=weight)


def load_fp32(seed, fold, project=None):
    # project 有給時(2026-11-10 補種子): 讀 compare_results/<project>/weights/ 的重訓 FP32 權重。
    #   主結果種子 1–4 的權重清磁碟時刪了,重訓到另一個資料夾(切分相同,folds.csv 逐位元組相同),不冒用主結果的名稱。
    if project is None:
        return load_float(seed, fold)
    net = qat_fx.build_model({**qat_fx.CONFIGS[qat_fx.UNET_TAG], "pretrained": False})
    suffix = f"_s{seed}" if seed else ""
    path = qat_fx.ROOT / "compare_results" / project / "weights" / f"{qat_fx.UNET_TAG}{suffix}_fold{fold}.pth"
    net.load_state_dict(torch.load(path, map_location="cpu")["model_state_dict"])
    return net.eval()


def prepare(seed, fold, w_bits, a_bits, sparse_init=None, project=None, suffix=""):
    qcfg = QConfigMapping().set_global(qconfig(w_bits, a_bits))
    net = load_fp32(seed, fold, project)
    if sparse_init:
        # 2:4 + 量化(2026-11-10 第三輪): 起點改成 sparse24.py 存下的稀疏 FP32 權重(例如 sp24d_ft<suffix>_fold<k>.pth)
        net.load_state_dict(torch.load(OUT / "models" / f"{sparse_init}{suffix}_fold{fold}.pth", map_location="cpu"))
    return prepare_qat_fx(QNet(net).train(), qcfg, (torch.zeros(1, 1, *SIZE),)).to(DEVICE)


def zero_masks(model):
    # 每個 4 維 weight 記下非零位置;QAT 每步之後乘回去,讓剪掉的權重保持 0(dense 層幾乎沒有剛好是 0 的權重,遮罩等於全 1)
    return {n: (p != 0).float() for n, p in model.named_parameters() if p.dim() == 4}


def apply_zero_masks(model, masks):
    params = dict(model.named_parameters())
    with torch.no_grad():
        for n, m in masks.items():
            params[n].mul_(m)


def save_masks_fq(model, variant, fold, sets):
    model.eval()
    model.apply(disable_observer)
    for dataset, paths in sets.items():
        folder = OUT / "masks" / dataset / variant / f"fold{fold}"
        folder.mkdir(parents=True, exist_ok=True)
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            h, w = image.shape
            with torch.inference_mode():
                logits = model(torch.from_numpy(unet_input(image, SIZE)).to(DEVICE)).float()
            pred = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False).argmax(1)[0].cpu().numpy()
            code = np.zeros((h, w), dtype=np.uint8)
            for c in NAMES:
                code[pred == c + 1] = 1 << c
            cv2.imwrite(str(folder / f"{path.stem}.png"), code)


def eval_sets(fold, limit):
    return {"cv": read_split(f"fold{fold}_val")[:limit], "test": read_split("test")[:limit]}


def control(fold, args):
    # INT8 QAT 狀態走 fake-quant 路徑,確認評估方式不影響結果
    done = OUT / "models" / f"tq_qat_fq_fold{fold}.done"
    if done.exists():
        return
    prepared = prepare(0, fold, 8, 8)
    prepared.load_state_dict(torch.load(OUT / "models" / f"tq_qat_fold{fold}.pth", map_location=DEVICE))
    save_masks_fq(prepared, "tq_qat_fq", fold, eval_sets(fold, args.limit))
    done.touch()
    print(f"[control fold{fold}] tq_qat_fq 完成", flush=True)
    del prepared
    torch.cuda.empty_cache()


def one(seed, fold, args):
    # --init-seed 有給時: FP32 起點固定用該種子的主結果權重,seed 只當 QAT 的隨機種子(批次順序、增強);
    #   變體名稱改成 <tag>_qat_qs<seed>(遮罩放在種子 0 的位置,因為起點是種子 0),只量 QAT 本身的變異。
    #   校正影像固定(calib_paths 用 Random(0)),所以只校正的結果與 --init-seed 的種子相同,不另存。
    tag = f"tq_w{args.w_bits}a{args.a_bits}" + (f"_{args.sparse_init}" if args.sparse_init else "")
    init = seed if args.init_seed is None else args.init_seed
    suffix = (f"_s{seed}" if seed else "") if args.init_seed is None else f"_qs{seed}"
    if args.fp32_project:
        suffix = "_rt" + suffix
    done = OUT / "models" / f"{tag}{suffix}_fold{fold}.done"
    if done.exists():
        return
    if not args.fp32_project:
        check_same(init, fold)
    start = time.time()
    torch.manual_seed(seed)
    prepared = prepare(init, fold, args.w_bits, args.a_bits, args.sparse_init, args.fp32_project, suffix)
    masks = zero_masks(prepared) if args.sparse_init else {}
    if masks:
        kept = sum(m.sum().item() for m in masks.values()) / sum(m.numel() for m in masks.values())
        print(f"[{tag} s{seed} fold{fold}] 稀疏起點: 卷積權重非零比例 {kept:.1%}", flush=True)
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
            apply_zero_masks(prepared, masks)
        scheduler.step()
        prepared.apply(disable_observer)
        dice = val_dice(prepared, val_samples, SIZE)
        log.append((epoch, dice))
        if dice > best_dice:
            best_dice, best_state = dice, {k: v.detach().cpu().clone() for k, v in prepared.state_dict().items()}
    print(f"[{tag} s{seed} fold{fold}] val dice PTQ {log[0][1]:.4f} → 最佳 {best_dice:.4f};訓練 {time.time() - start:.0f} s",
          flush=True)

    sets = eval_sets(fold, args.limit)
    torch.save(best_state, OUT / "models" / f"{tag}_qat{suffix}_fold{fold}.pth")
    if args.init_seed is None:
        prepared.load_state_dict(ptq_state)
        save_masks_fq(prepared, f"{tag}_ptq{suffix}", fold, sets)
    prepared.load_state_dict(best_state)
    save_masks_fq(prepared, f"{tag}_qat{suffix}", fold, sets)
    with open(OUT / f"{tag}_val_log.csv", "a", newline="") as f:
        csv.writer(f).writerows((seed, fold, e, round(d, 5)) for e, d in log)
    done.touch()
    print(f"[{tag} s{seed} fold{fold}] 完成,累計 {time.time() - start:.0f} s", flush=True)
    del prepared
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--w-bits", type=int, default=4)
    parser.add_argument("--a-bits", type=int, default=4)
    parser.add_argument("--calib", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fp32-project", default=None, help="FP32 起點改讀 compare_results/<這個>/weights/(重訓的種子);輸出名稱加 _rt")
    parser.add_argument("--sparse-init", default=None, help="稀疏 FP32 起點的權重前綴(例如 sp24d_ft),QAT 時保持稀疏")
    parser.add_argument("--init-seed", type=int, default=None, help="FP32 起點固定用這個種子,--seeds 只當 QAT 種子")
    parser.add_argument("--control", action="store_true", help="只跑 INT8 QAT 狀態的 fake-quant 對照")
    args = parser.parse_args()
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    for fold in args.folds:
        if args.control:
            control(fold, args)
            continue
        for seed in args.seeds:
            one(seed, fold, args)


if __name__ == "__main__":
    main()
