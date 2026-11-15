import argparse
import copy
import csv
import time
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stage2.compare_yolo_unet import GT_DIR, NAMES, ROOT, load_gt
from deployment.int8_quant import read_split, unet_input
from stage2.kfold_pano import EXCLUDE_TEST
from stage2.mf_postproc import largest, mf_metrics
from analysis.paper_metrics import pixel_metrics
from hardware.qat_fx import OUT, SIZE, convert, requant
from hardware.qat_int4 import prepare

warnings.filterwarnings("ignore")
# 2026-11-10 定點 mf Dice 掉 0.004 的原因(使用者要求;見 EXPERIMENTS.md「補完整流程與種子」)。伺服器 CPU,種子 0,測試集。
#   dense(tq_qat_fold<k>.pth)與 2:4(tq_w8a8_sp24d_ft_qat_fold<k>.pth)各自:
#   1. 整數參考: convert_fx 分別在 fbgemm 與 x86 引擎下轉(qat_fx.py 的 dense 用 fbgemm;sparse_int8_verify.py 沒設,用了預設 x86);
#   2. 定點變體(和 qat_fx.FxConv / fx_add 相同的算法,逐項關掉一個捨入來源):
#      fx = 原設定(16 位元乘數、偏置四捨五入成整數、加法兩邊各自 requant 再相加);m31 = 31 位元乘數;fbias = 偏置不捨入;
#      add1 = 加法先相加再一次 requant;all = 三項都關;
#   3. 逐層局部誤差: 把整數模型每層卷積的實際輸入餵給定點卷積,和整數輸出比(不含前面層的累積);
#   4. 端點: 每個變體對整數參考的像素不一致比例、mf 遮罩 Dice,以及對 GT 的 mf / ian Dice(同 slim_eval: mf 取最大連通區塊)。


class FxConv(nn.Module):
    def __init__(self, q, bits, round_bias):
        super().__init__()
        self.q, self.bits, self.round_bias = q, bits, round_bias
        self.relu = "ReLU" in type(q).__name__

    def forward(self, x):
        q = self.q
        w = q.weight()
        ws = w.q_per_channel_scales().double().view(1, -1, 1, 1)
        si, zi = x.q_scale(), x.q_zero_point()
        acc = F.conv2d(x.int_repr().double() - zi, w.int_repr().double(), None, q.stride, q.padding, q.dilation, q.groups)
        if q.bias() is not None:
            b = q.bias().double().view(1, -1, 1, 1) / (si * ws)
            acc = acc + (torch.round(b) if self.round_bias else b)
        acc = acc.clamp(-2.0 ** 31, 2.0 ** 31 - 1)
        y = requant(acc, si * ws / q.scale, self.bits) + q.zero_point
        low = q.zero_point if self.relu else 0
        return torch._make_per_tensor_quantized_tensor(y.clamp(low, 255).to(torch.uint8), q.scale, q.zero_point)


def fx_add(bits, relu, single):
    def add(a, b, scale, zero_point):
        da, db = a.int_repr().double() - a.q_zero_point(), b.int_repr().double() - b.q_zero_point()
        if single:
            y = torch.floor((da * a.q_scale() + db * b.q_scale()) / scale + 0.5) + zero_point
        else:
            y = requant(da, a.q_scale() / scale, bits) + requant(db, b.q_scale() / scale, bits) + zero_point
        return torch._make_per_tensor_quantized_tensor(y.clamp(zero_point if relu else 0, 255).to(torch.uint8), scale, zero_point)
    return add


def to_fixed(model, bits=16, round_bias=True, single_add=False):
    model = copy.deepcopy(model)
    for name, module in list(model.named_modules()):
        if "Conv2d" in type(module).__name__ or "ConvReLU2d" in type(module).__name__:
            parent = model.get_submodule(name.rpartition(".")[0]) if "." in name else model
            setattr(parent, name.rpartition(".")[2], FxConv(module, bits, round_bias))
    for node in model.graph.nodes:
        if node.op == "call_function" and getattr(node.target, "__name__", "") in ("add", "add_relu") and "quantized" in str(node.target):
            node.target = fx_add(bits, node.target.__name__ == "add_relu", single_add)
    model.recompile()
    return model


VARIANTS = {"fx": (16, True, False), "m31": (31, True, False), "fbias": (16, False, False), "add1": (16, True, True), "all": (31, False, True)}


def convs(model):
    return {n: m for n, m in model.named_modules() if "Conv2d" in type(m).__name__ or "ConvReLU2d" in type(m).__name__}


def local_errors(qmodel, images):
    # 每層: 定點(16 位元、偏置捨入)對整數輸出差幾個 LSB;另記 requant 乘數 M 的 16 位元相對誤差最大值、偏置捨入量(LSB)
    stats, hooks = {}, []
    for name, m in convs(qmodel).items():
        fx = FxConv(m, 16, True)
        fxb = FxConv(m, 16, False)

        def hook(mod, inp, out, name=name, fx=fx, fxb=fxb):
            a, b, c = out.int_repr().int(), fx(inp[0]).int_repr().int(), fxb(inp[0]).int_repr().int()
            s = stats.setdefault(name, [0, 0, 0, 0.0, 0])
            s[0] += (a != b).sum().item()
            s[1] += a.numel()
            s[2] += (a != c).sum().item()
            s[3] += (b - a).double().sum().item()
            s[4] = max(s[4], (b - a).abs().max().item())
        hooks.append(m.register_forward_hook(hook))
    with torch.inference_mode():
        for x in images:
            qmodel(x)
    for h in hooks:
        h.remove()
    rows = []
    for name, m in convs(qmodel).items():
        w = m.weight()
        ws = w.q_per_channel_scales().double()
        rows.append((name, m.scale, int((w.int_repr() != 0).float().mean().item() * 1000) / 1000,
                     float(ws.min()), float(ws.max()), *stats[name][:1], stats[name][1], stats[name][2],
                     stats[name][3] / stats[name][1], stats[name][4]))
    return rows


def predict(model, image):
    h, w = image.shape
    with torch.inference_mode():
        logits = model(torch.from_numpy(unet_input(image, SIZE)))
    return F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False).argmax(1)[0].numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--local", type=int, default=8)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    paths = [p for p in read_split("test") if p.stem not in EXCLUDE_TEST][:args.limit]
    images = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]
    gts = [load_gt(p.stem, im.shape, (ROOT / GT_DIR).resolve(), {}) for p, im in zip(paths, images)]
    out = OUT / "fx_diag"
    out.mkdir(exist_ok=True)
    end_rows, layer_rows = [], []
    for fold in args.folds:
        for kind, init, state in (("dense", None, "tq_qat"), ("sp24", "sp24d_ft", "tq_w8a8_sp24d_ft_qat")):
            start = time.time()
            st = torch.load(OUT / "models" / f"{state}_fold{fold}.pth", map_location="cpu")
            models = {}
            for engine in ("fbgemm", "x86"):
                torch.backends.quantized.engine = engine
                models[f"int_{engine}"] = convert(st, prepare(0, fold, 8, 8, init).cpu())
            torch.backends.quantized.engine = "fbgemm"
            ref = models["int_fbgemm"]
            for v, (bits, rb, single) in VARIANTS.items():
                models[v] = to_fixed(ref, bits, rb, single)
            x8 = [torch.from_numpy(unet_input(im, SIZE)) for im in images[:args.local]]
            layer_rows += [(fold, kind, *r) for r in local_errors(ref, x8)]
            preds = {}
            for v, m in models.items():
                torch.backends.quantized.engine = "x86" if v == "int_x86" else "fbgemm"
                preds[v] = [predict(m, im) for im in images]
            torch.backends.quantized.engine = "fbgemm"
            for v, ps in preds.items():
                dis, agree, mfd, iand = [], [], [], []
                for p, r, gt in zip(ps, preds["int_fbgemm"], gts):
                    dis.append(float((p != r).mean()))
                    agree.append(pixel_metrics(p == 1, r == 1)["dice"])
                    mfd.append(mf_metrics(largest(p == 1), gt["mf"])["mf_dice"] if "mf_dice" in mf_metrics(largest(p == 1), gt["mf"]) else np.nan)
                    iand.append(pixel_metrics(p == 2, gt["ian"])["dice"])
                end_rows.append((fold, kind, v, np.mean(dis), np.nanmean(agree), np.nanmean(mfd), np.nanmean(iand)))
                print(f"[fold{fold} {kind:5s} {v:10s}] 像素不一致 {np.mean(dis):.2e} mf 對整數 Dice {np.nanmean(agree):.4f} "
                      f"mf Dice {np.nanmean(mfd):.4f} ian Dice {np.nanmean(iand):.4f}", flush=True)
            print(f"[fold{fold} {kind}] {time.time() - start:.0f} s", flush=True)
            with open(out / "end.csv", "w", newline="") as f:
                csv.writer(f).writerows([("fold", "kind", "variant", "pixel_disagree", "mf_agree_dice", "mf_dice", "ian_dice")] + end_rows)
            with open(out / "layers.csv", "w", newline="") as f:
                csv.writer(f).writerows([("fold", "kind", "layer", "out_scale", "nonzero", "ws_min", "ws_max", "diff_n", "n",
                                          "diff_n_floatbias", "mean_signed_lsb", "max_abs_lsb")] + layer_rows)


if __name__ == "__main__":
    main()
