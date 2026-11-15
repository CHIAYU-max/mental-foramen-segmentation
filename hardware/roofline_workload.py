import argparse
import csv
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 工作負載特性與加速器需求範圍(2026-10-26):
#   只用 roofline_profile.py 已存的逐層資料(compare_results/roofline/<model>/layers.csv),不重新量測、不需要 GPU。
#   假設 BatchNorm / ReLU / SiLU 等逐元素運算都融合進前一個卷積(部署時的常見做法),所以只計卷積層:
#     MAC = flops / 2;權重 = params 個;activation = 輸入與輸出元素數;INT8 = 1 byte / 元素、FP16 = 2 bytes。
#   每層「工作集」= 權重 + 輸入 + 輸出(整層不切塊時片上要放得下的量);運算強度 = MAC / 工作集 bytes。
#   DRAM 流量兩種極端: 逐層(每層都從 DRAM 讀輸入與權重、寫回輸出)與理想(只讀權重一次、輸入、輸出)。
#   需求: 給定延遲目標,在 100% 利用率下需要的 MAC/s、PE 數(依時脈)與 DRAM 頻寬。真實利用率會更低,所以是下限。
#   平台參考值: Basys3 XC7A35T 有 90 個 DSP48E1,100 MHz、每 DSP 每週期 1 MAC 約 9 GMAC/s;
#   RTX 3090 用 roofline_profile 實測的 23.9 TFLOP/s(FP32,約 11.9 TMAC/s)與 865 GB/s。
MODELS = {
    "stage2_unet_r34": "U-Net R34 (512x512)",
    "stage2_yolov8n-seg_fused": "YOLOv8n-seg (1280x896, fused)",
    "stage1_teeth_yolov8s_fused": "Stage 1 YOLOv8s-seg (1024x544, fused)",
}
TARGETS_MS = [10, 50, 100, 500]
CLOCKS_MHZ = [100, 500, 1000]
BASYS3_GMACS = 90 * 100e6 / 1e9
RTX3090_GMACS = 23887 / 2


def prod(shape):
    return math.prod(int(v) for v in shape.split("x"))


def analyze(model_dir):
    with open(model_dir / "layers.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    convs = []
    for r in rows:
        if not r["type"].startswith("Conv"):
            continue
        mac = float(r["flops"]) / 2
        w, a_in, a_out = int(r["params"]), prod(r["in_shape"]), prod(r["out_shape"])
        convs.append({"seq": int(r["seq"]), "name": r["name"], "type": r["type"], "in_shape": r["in_shape"],
                      "out_shape": r["out_shape"], "mac": mac, "weights": w, "act_in": a_in, "act_out": a_out,
                      "ws_int8": w + a_in + a_out, "oi_int8": mac / (w + a_in + a_out)})
    total_mac = sum(c["mac"] for c in convs)
    all_flops = sum(float(r["flops"]) for r in rows)
    conv_flops = sum(float(r["flops"]) for r in rows if r["type"].startswith("Conv"))
    first, last = convs[0], convs[-1]
    s = {
        "layers": len(convs),
        "gmac": total_mac / 1e9,
        "weights_mb_int8": sum(c["weights"] for c in convs) / 1e6,
        "conv_flop_share": conv_flops / all_flops,
        "max_ws_mb_int8": max(c["ws_int8"] for c in convs) / 1e6,
        "max_ws_layer": max(convs, key=lambda c: c["ws_int8"])["name"],
        "max_act_mb_int8": max(max(c["act_in"], c["act_out"]) for c in convs) / 1e6,
        "traffic_layerwise_mb_int8": sum(c["ws_int8"] for c in convs) / 1e6,
        "traffic_ideal_mb_int8": (sum(c["weights"] for c in convs) + first["act_in"] + last["act_out"]) / 1e6,
        "top3_mac_share": sum(sorted((c["mac"] for c in convs), reverse=True)[:3]) / total_mac,
    }
    s["oi_layerwise_int8"] = total_mac / (s["traffic_layerwise_mb_int8"] * 1e6)
    s["oi_ideal_int8"] = total_mac / (s["traffic_ideal_mb_int8"] * 1e6)
    # U-Net: 編碼器輸出要留到解碼器(skip connection),片上或 DRAM 都得放
    if "unet" in model_dir.name:
        last_of = {}
        for c in convs:
            m = re.match(r"(stem|layers\.\d)", c["name"])
            if m:
                last_of[m.group(1)] = c
        skips = [last_of[k] for k in ("stem", "layers.0", "layers.1", "layers.2") if k in last_of]
        s["skip_mb_int8"] = sum(c["act_out"] for c in skips) / 1e6
    summary_md = model_dir / "summary.md"
    if summary_md.exists():
        m = re.search(r"memory-bound layers: \*?\*?([\d.]+)%", summary_md.read_text())
        s["measured_mem_bound_share"] = float(m.group(1)) / 100 if m else float("nan")
    return convs, s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    args = parser.parse_args()
    base = ROOT / "compare_results" / "roofline"
    out = base / "workload"
    out.mkdir(exist_ok=True)

    lines = ["# 工作負載特性與加速器需求範圍", "",
             "由 `roofline_workload.py` 產生;假設逐元素運算融合進卷積,只計卷積層;INT8 = 1 byte / 元素。", ""]
    summaries = {}
    for name in args.models:
        convs, s = analyze(base / name)
        summaries[name] = s
        with open(out / f"{name}_layers.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(convs[0]))
            writer.writeheader()
            writer.writerows(convs)

    lines += ["## 工作負載", "",
              "| 模型 | 卷積層 | GMAC | 權重 MB | 卷積佔 FLOPs | 前 3 層佔 MAC | 最大工作集 MB(層) | 最大 activation MB | skip MB | 流量 逐層 / 理想 MB | 強度 逐層 / 理想 MAC/B | 實測 memory-bound 時間佔比(3090) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, s in summaries.items():
        lines.append(f"| {MODELS.get(name, name)} | {s['layers']} | {s['gmac']:.2f} | {s['weights_mb_int8']:.2f} | {s['conv_flop_share']:.1%} | "
                     f"{s['top3_mac_share']:.1%} | {s['max_ws_mb_int8']:.2f}({s['max_ws_layer']}) | {s['max_act_mb_int8']:.2f} | "
                     f"{s.get('skip_mb_int8', float('nan')):.2f} | {s['traffic_layerwise_mb_int8']:.1f} / {s['traffic_ideal_mb_int8']:.1f} | "
                     f"{s['oi_layerwise_int8']:.0f} / {s['oi_ideal_int8']:.0f} | {s.get('measured_mem_bound_share', float('nan')):.1%} |")

    lines += ["", "## 需求範圍(100% 利用率下的下限)", "",
              "| 模型 | 延遲目標 ms | 需要 GMAC/s | PE 數 @100 / 500 / 1000 MHz | DRAM 頻寬 逐層 / 理想 GB/s |",
              "| --- | --- | --- | --- | --- |"]
    for name, s in summaries.items():
        for t in TARGETS_MS:
            gmacs = s["gmac"] / (t / 1000)
            pes = " / ".join(f"{math.ceil(gmacs * 1e9 / (mhz * 1e6)):,}" for mhz in CLOCKS_MHZ)
            lines.append(f"| {MODELS.get(name, name)} | {t} | {gmacs:,.0f} | {pes} | "
                         f"{s['traffic_layerwise_mb_int8'] / t:.1f} / {s['traffic_ideal_mb_int8'] / t:.1f} |")

    lines += ["", "## 平台對照(計算下限,不含搬移)", "",
              "| 模型 | Basys3 90 DSP @100 MHz(約 9 GMAC/s) | RTX 3090 實測峰值(約 11.9 TMAC/s) |", "| --- | --- | --- |"]
    for name, s in summaries.items():
        lines.append(f"| {MODELS.get(name, name)} | {s['gmac'] / BASYS3_GMACS * 1000:,.0f} ms | {s['gmac'] / RTX3090_GMACS * 1000:.2f} ms |")

    (out / "workload_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # U-Net 逐層: MAC 與工作集沿著網路深度(論文 / SOP 用)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    convs, _ = analyze(base / "stage2_unet_r34")
    x = range(len(convs))
    fig, ax1 = plt.subplots(figsize=(8, 3.4))
    ax1.bar(x, [c["mac"] / 1e9 for c in convs], color="#4c72b0", label="GMAC per layer")
    ax1.set_ylabel("GMAC")
    ax1.set_xlabel("Convolution layer (input to output)")
    ax2 = ax1.twinx()
    ax2.plot(x, [c["ws_int8"] / 1e6 for c in convs], color="#dd8452", marker=".", label="Working set, INT8 (MB)")
    ax2.set_ylabel("Working set (MB, INT8)")
    for label, idx in (("encoder", 0), ("decoder", next(i for i, c in enumerate(convs) if c["name"].startswith("decoders"))),
                       ("head", next(i for i, c in enumerate(convs) if c["name"].startswith("head")))):
        ax1.axvline(idx - 0.5, color="gray", lw=0.6, ls=":")
        ax1.text(idx, 1.02, label, fontsize=8, color="gray", transform=ax1.get_xaxis_transform())
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "unet_layers.png", dpi=200)
    print(f"\n輸出: {out}")


if __name__ == "__main__":
    main()
