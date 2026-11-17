"""2026-11-16: 把 unet_perf_count.py 的逐節點時間對回 PyTorch 的卷積層,算每層運算量、實際 GMAC/s 與 INT8 / FP32 倍數。

只用合成影像的剖析結果與模型形狀(不需要病人影像)。輸出 compare_results/perf_count/<主機>/layers.csv、layers.md
"""
import csv
import re
import platform
from collections import defaultdict

import torch

from stage2.compare_yolo_unet import ROOT
from stage2.kfold_unet_variants import CONFIGS, build_model

D = ROOT / "compare_results" / "perf_count" / platform.node()

# PyTorch 端: 每個卷積的形狀與 MAC(R34/512,輸入 1×1×512×512)
model = build_model({**CONFIGS["r34_imagenet_aug"], "pretrained": False}).eval()
info = {}


def hook(name):
    def f(m, x, y):
        k = m.kernel_size[0] * m.kernel_size[1]
        info[name] = dict(cin=m.in_channels, cout=m.out_channels, k=m.kernel_size[0], stride=m.stride[0],
                          h=y.shape[2], w=y.shape[3], gmac=y.numel() * m.in_channels // m.groups * k / 1e9)
    return f


for n, m in model.named_modules():
    if isinstance(m, torch.nn.Conv2d):
        m.register_forward_hook(hook(n))
with torch.no_grad():
    model(torch.zeros(1, 1, 512, 512))


def conv_of(original):
    """OpenVINO 節點的 originalLayersNames → PyTorch 卷積名稱(由融合進來的 BN 推回)。"""
    for s in original.split(","):
        m = re.match(r"__module\.(.+?)/aten::_?convolution", s)
        if m:
            return m.group(1)
    for s in original.split(","):
        m = re.match(r"__module\.(.+?)/aten::batch_norm", s)
        if m:
            p = m.group(1)
            p = re.sub(r"\.bn(\d)$", r".conv\1", p)
            p = re.sub(r"downsample\.1$", "downsample.0", p)
            p = re.sub(r"^stem\.1$", "stem.0", p)
            p = re.sub(r"block\.(\d)$", lambda g: f"block.{int(g.group(1)) - 1}", p)
            return p
    return None


def load(tag):
    out, rest = {}, defaultdict(float)
    for r in csv.DictReader(open(D / f"nodes_{tag}.csv", encoding="utf-8-sig")):
        ms = float(r["ms"])
        if r["layer_type"] == "Convolution":
            out[conv_of(r["original"]) or "?"] = ms
        elif ms > 0:
            rest[r["layer_type"]] += ms
    return out, rest


i8, rest8 = load("int8")
f32, rest32 = load("fp32")
unmatched = sorted(set(info) - set(i8))
assert len(unmatched) <= 1, unmatched  # 只剩輸出頭 head.1(沒有 BN)
if unmatched:
    i8[unmatched[0]], f32[unmatched[0]] = i8.pop("?"), f32.pop("?")


def block(name):
    if name.startswith("stem"):
        return "encoder stem"
    if name.startswith("layers"):
        return f"encoder layer{int(name.split('.')[1]) + 1}"
    if name.startswith("decoders"):
        return f"decoder {name.split('.')[1]}"
    return "head"


rows = []
for n, s in info.items():
    rows.append(dict(layer=n, block=block(n), cin=s["cin"], cout=s["cout"], k=s["k"], out=f"{s['h']}x{s['w']}",
                     gmac=round(s["gmac"], 3), int8_ms=i8[n], fp32_ms=f32[n],
                     int8_gmacs=round(s["gmac"] / i8[n] * 1e3, 1), fp32_gmacs=round(s["gmac"] / f32[n] * 1e3, 1),
                     speedup=round(f32[n] / i8[n], 2)))
with open(D / "layers.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

tot = dict(gmac=sum(r["gmac"] for r in rows), i8=sum(i8.values()) + sum(rest8.values()), f32=sum(f32.values()) + sum(rest32.values()))
L = [f"# U-Net R34/512 逐層: 運算量、實際速度、INT8 / FP32(筆電 i5-13500H,OpenVINO CPU PERF_COUNT,合成影像)", "",
     f"- 卷積 {len(rows)} 層、{tot['gmac']:.1f} GMAC;逐節點加總 INT8 {tot['i8']:.1f} ms、FP32 {tot['f32']:.1f} ms(同一次量測內比較)", "",
     "## 依區塊", "", "| 區塊 | 層數 | GMAC | GMAC 佔比 | INT8 ms | INT8 時間佔比 | INT8 GMAC/s | FP32 ms | FP32 GMAC/s | INT8 快幾倍 |",
     "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
order = ["encoder stem"] + [f"encoder layer{i}" for i in range(1, 5)] + [f"decoder {i}" for i in range(4)] + ["head"]
for b in order:
    rs = [r for r in rows if r["block"] == b]
    g, a, c = sum(r["gmac"] for r in rs), sum(r["int8_ms"] for r in rs), sum(r["fp32_ms"] for r in rs)
    L.append(f"| {b} | {len(rs)} | {g:.2f} | {g / tot['gmac']:.1%} | {a:.2f} | {a / tot['i8']:.1%} | {g / a * 1e3:.0f} | {c:.2f} | {g / c * 1e3:.0f} | {c / a:.2f} |")
L += ["", "非卷積(ms): INT8 " + "、".join(f"{k} {v:.2f}" for k, v in sorted(rest8.items(), key=lambda kv: -kv[1]) if v > 0.05)
      + ";FP32 " + "、".join(f"{k} {v:.2f}" for k, v in sorted(rest32.items(), key=lambda kv: -kv[1]) if v > 0.05), "",
      "## INT8 最花時間的 12 層", "", "| 層 | in→out | k | 輸出 | GMAC | INT8 ms | INT8 GMAC/s | FP32 GMAC/s | INT8 快幾倍 |",
      "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
for r in sorted(rows, key=lambda r: -r["int8_ms"])[:12]:
    L.append(f"| {r['layer']} | {r['cin']}→{r['cout']} | {r['k']} | {r['out']} | {r['gmac']:.2f} | {r['int8_ms']:.2f} | {r['int8_gmacs']:.0f} | {r['fp32_gmacs']:.0f} | {r['speedup']:.2f} |")
L += ["", "## INT8 效率最低的 8 層(GMAC/s,只列 ≥ 0.1 ms)", "", "| 層 | in→out | k | 輸出 | GMAC | INT8 ms | INT8 GMAC/s | INT8 快幾倍 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
for r in sorted([r for r in rows if r["int8_ms"] >= 0.1], key=lambda r: r["int8_gmacs"])[:8]:
    L.append(f"| {r['layer']} | {r['cin']}→{r['cout']} | {r['k']} | {r['out']} | {r['gmac']:.3f} | {r['int8_ms']:.2f} | {r['int8_gmacs']:.0f} | {r['speedup']:.2f} |")
(D / "layers.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
