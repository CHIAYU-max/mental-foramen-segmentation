import argparse
import copy
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from deployment.int8_quant import read_split
from hardware.qat_fx import OUT, convert, requant, save_masks
from hardware.qat_int4 import prepare

warnings.filterwarnings("ignore")
# 2026-11-11 定點設計的五種子、五折驗證(使用者要求「跑完完整種子訓練和折數,確定可以用之後再來加入」;見 EXPERIMENTS.md「補完整流程與種子」)。
#   fx_diag.py(種子 0、只看測試)發現 2:4 定點 mf Dice −0.004 來自殘差加法兩次捨入與偏置捨入。這裡把建議的 RTL 做法寫成具體規格,
#   和整數模型(fbgemm)、原定點設定(qat_fx.to_fixed)比,種子 0–4、五折、CV + 測試都存遮罩,之後用 slim_eval 配對比較:
#     fxs_<kind>_int: convert_fx 整數模型(fbgemm 引擎;sparse_int8_verify.py 用的是預設 x86,所以重存一份)
#     fxs_<kind>_fx : 原設定(32 位元累加器飽和、16 位元乘數、偏置捨入成整數、殘差加法兩邊各自 requant 成整數再相加)
#     fxs_<kind>_rtl: 建議規格(16 位元乘數;偏置與殘差加法都多留 F = 8 個小數位元,最後只捨入一次)
#       卷積: acc(32 位元飽和)× 2^F + round(bias × 2^F),再以 M = si·ws/so 的 16 位元乘數 requant(右移多 F 位);
#       加法: 兩個輸入各自用 16 位元乘數 requant 到「輸出 scale × 2^-F」(四捨五入),相加後右移 F 位四捨五入一次。
#   kind: dense = tq_qat(種子 0)/ tq_w8a8_qat_rt_s<k>;sp24 = tq_w8a8_sp24d_ft_qat(種子 0)/ tq_w8a8_sp24d_ft_qat_rt_s<k>。
#   遮罩: compare_results/int8_pano/masks/{cv,test}/fxs_<kind>_<variant>[_s<k>]/fold<f>/;每個 (種子, 種類, 折) 寫 .done,可續跑。
RT = "kfold_pano_mf_559_rt"
FRAC = 8


def requant_frac(x, multiplier, bits, frac):
    # 和 qat_fx.requant 相同的 16 位元乘數,但多保留 frac 個小數位元(四捨五入到 2^-frac)
    multiplier = torch.as_tensor(multiplier, dtype=torch.float64)
    exponent = torch.floor(torch.log2(multiplier)) + 1
    m = torch.round(multiplier * 2.0 ** (bits - exponent))
    shift = bits - exponent - frac
    return torch.floor(x * m / 2.0 ** shift + 0.5)


class RtlConv(nn.Module):
    def __init__(self, q, bits=16, frac=FRAC):
        super().__init__()
        self.q, self.bits, self.frac = q, bits, frac
        self.relu = "ReLU" in type(q).__name__

    def forward(self, x):
        q = self.q
        w = q.weight()
        ws = w.q_per_channel_scales().double().view(1, -1, 1, 1)
        si, zi = x.q_scale(), x.q_zero_point()
        acc = F.conv2d(x.int_repr().double() - zi, w.int_repr().double(), None, q.stride, q.padding, q.dilation, q.groups)
        acc = acc.clamp(-2.0 ** 31, 2.0 ** 31 - 1) * 2.0 ** self.frac
        if q.bias() is not None:
            acc = acc + torch.round(q.bias().double().view(1, -1, 1, 1) / (si * ws) * 2.0 ** self.frac)
        y = requant(acc, si * ws / q.scale / 2.0 ** self.frac, self.bits) + q.zero_point
        low = q.zero_point if self.relu else 0
        return torch._make_per_tensor_quantized_tensor(y.clamp(low, 255).to(torch.uint8), q.scale, q.zero_point)


def rtl_add(bits, relu, frac=FRAC):
    def add(a, b, scale, zero_point):
        s = requant_frac(a.int_repr().double() - a.q_zero_point(), a.q_scale() / scale, bits, frac) \
            + requant_frac(b.int_repr().double() - b.q_zero_point(), b.q_scale() / scale, bits, frac)
        y = torch.floor(s / 2.0 ** frac + 0.5) + zero_point
        return torch._make_per_tensor_quantized_tensor(y.clamp(zero_point if relu else 0, 255).to(torch.uint8), scale, zero_point)
    return add


def to_rtl(model, bits=16):
    model = copy.deepcopy(model)
    convs = adds = 0
    for name, module in list(model.named_modules()):
        if "Conv2d" in type(module).__name__ or "ConvReLU2d" in type(module).__name__:
            parent = model.get_submodule(name.rpartition(".")[0]) if "." in name else model
            setattr(parent, name.rpartition(".")[2], RtlConv(module, bits))
            convs += 1
    for node in model.graph.nodes:
        if node.op == "call_function" and getattr(node.target, "__name__", "") in ("add", "add_relu") and "quantized" in str(node.target):
            node.target = rtl_add(bits, node.target.__name__ == "add_relu")
            adds += 1
    model.recompile()
    print(f"RTL 規格替換: 卷積 {convs}、加法 {adds}", flush=True)
    return model


def load(seed, fold, kind):
    if kind == "dense":
        state = "tq_qat" if seed == 0 else f"tq_w8a8_qat_rt_s{seed}"
        prepared = prepare(seed, fold, 8, 8) if seed == 0 else prepare(seed, fold, 8, 8, None, RT)
    else:
        state = "tq_w8a8_sp24d_ft_qat" if seed == 0 else f"tq_w8a8_sp24d_ft_qat_rt_s{seed}"
        prepared = prepare(0, fold, 8, 8, "sp24d_ft") if seed == 0 else prepare(seed, fold, 8, 8, "sp24d_ft", RT, f"_rt_s{seed}")
    st = torch.load(OUT / "models" / f"{state}_fold{fold}.pth", map_location="cpu")
    return convert(st, prepared.cpu())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--kind", choices=["dense", "sp24"], required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.backends.quantized.engine = "fbgemm"
    import hardware.qat_fx as qat_fx
    suffix = f"_s{args.seed}" if args.seed else ""
    for fold in args.folds:
        done = OUT / "models" / f"fxs_{args.kind}{suffix}_fold{fold}.done"
        if done.exists():
            continue
        start = time.time()
        q = load(args.seed, fold, args.kind)
        sets = {"cv": read_split(f"fold{fold}_val")[:args.limit], "test": read_split("test")[:args.limit]}
        for variant, model in (("int", q), ("fx", qat_fx.to_fixed(q, 32, 16)), ("rtl", to_rtl(q))):
            save_masks(model, f"fxs_{args.kind}_{variant}{suffix}", fold, sets)
        if args.limit is None:
            done.touch()
        print(f"[s{args.seed} {args.kind} fold{fold}] 完成 {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
