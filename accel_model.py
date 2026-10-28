import argparse
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 加速器分析模型(L2,2026-10-26):掃 PE 陣列大小 × 片上 SRAM,估每層的利用率、DRAM 流量與延遲。
#   輸入: roofline_workload.py 產生的 compare_results/roofline/workload/<model>_layers.csv(只含卷積層,
#         假設 BN / ReLU / SiLU 已融合)。不需要 GPU,也不是 RTL,是一階的解析模型,數字只用來比較趨勢。
#   假設(都可以用參數改):
#     1. 陣列: R × C 的 weight-stationary 脈動陣列,INT8,每 PE 每週期 1 MAC,時脈 --mhz(預設 500)。
#     2. 對映: 卷積看成 GEMM,M = 輸出像素數、K = 歸約長度(MAC / (M·N),自動涵蓋分組卷積)、N = 輸出通道。
#        K 對到 R 列、N 對到 C 行;每個權重塊要串流 M 個像素,外加 R + C 週期的填滿與排空:
#          cycles = ceil(K/R) · ceil(N/C) · (M + R + C)    utilization = MAC / (cycles · R · C)
#     3. 記憶體: 片上 SRAM S bytes,在 S 內挑輸出切塊 (Tm 像素 × Tn 通道) 讓 DRAM 流量最小:
#          片上需要 Tm·Cin(輸入列,忽略 halo)+ K·Tn(權重)+ Tm·Tn(輸出)≤ S
#          流量 = act_in · ceil(N/Tn) + weights · ceil(M/Tm) + act_out
#        整層放得下(工作集 ≤ S)時就是 act_in + weights + act_out。
#     4. 層間保留: 下一層的輸入就是這一層的輸出(空間大小相同、這層輸出 ≤ 下一層輸入)時,
#        若 這層輸出 + 下一層權重 + 下一層輸出 ≤ S,輸出就留在片上,省掉一次寫出與讀回(2 × act_out)。
#        層的順序沿用 layers.csv(大致是執行順序);ResNet 的 downsample、YOLO 的分支因形狀接不上而不會被算進來。
#     5. U-Net 的 skip(編碼器 4 個輸出,約 6 MB)若放不下(S < skip + 該模型最大工作集的一半),
#        就要寫出 DRAM 再讀回: 額外 2 × skip bytes。
#     6. 每層延遲 = max(cycles / f, 流量 / DRAM 頻寬);層與層之間不重疊。
#   輸出: compare_results/roofline/accel_model/(sweep.csv、summary.md、三張圖)。
MODELS = {"stage2_unet_r34": "U-Net R34 512x512", "stage2_yolov8n-seg_fused": "YOLOv8n-seg 1280x896"}
ARRAYS = [8, 16, 32, 64]
SRAM_KB = [128, 256, 512, 1024, 2048, 4096, 8192]
BW_GBS = [4.0, 12.8, 25.6]
SKIP_MB = {"stage2_unet_r34": 6.03}


def load_layers(name):
    with open(ROOT / "compare_results" / "roofline" / "workload" / f"{name}_layers.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    layers = []
    for r in rows:
        out = [int(v) for v in r["out_shape"].split("x")]
        inp = [int(v) for v in r["in_shape"].split("x")]
        cin = inp[1]
        m, n = out[2] * out[3], out[1]
        mac = float(r["mac"])
        layers.append({"name": r["name"], "M": m, "N": n, "K": round(mac / (m * n)), "Cin": cin, "mac": mac,
                       "in_hw": (inp[2], inp[3]), "out_hw": (out[2], out[3]),
                       "weights": int(r["weights"]), "act_in": int(r["act_in"]), "act_out": int(r["act_out"])})
    return layers


def compute_cycles(layer, rows, cols):
    return math.ceil(layer["K"] / rows) * math.ceil(layer["N"] / cols) * (layer["M"] + rows + cols)


def dram_traffic(layer, sram):
    if layer["weights"] + layer["act_in"] + layer["act_out"] <= sram:
        return layer["weights"] + layer["act_in"] + layer["act_out"]
    best = math.inf
    tm = 1
    while tm <= layer["M"]:
        for tn in sorted({min(layer["N"], 2 ** i) for i in range(0, 13)}):
            if tm * layer["Cin"] + layer["K"] * tn + tm * tn > sram:
                continue
            traffic = (layer["act_in"] * math.ceil(layer["N"] / tn) + layer["weights"] * math.ceil(layer["M"] / tm)
                       + layer["act_out"])
            best = min(best, traffic)
        tm *= 2
    return best


def evaluate(name, layers, array, sram_kb, bw, mhz):
    sram = sram_kb * 1024
    max_ws = max(l["weights"] + l["act_in"] + l["act_out"] for l in layers)
    total_cycles = total_traffic = latency = mac = 0.0
    compute_bound = 0
    for i, l in enumerate(layers):
        cyc = compute_cycles(l, array, array)
        traffic = dram_traffic(l, sram)
        nxt = layers[i + 1] if i + 1 < len(layers) else None
        if (nxt and nxt["in_hw"] == l["out_hw"] and l["act_out"] <= nxt["act_in"]
                and l["act_out"] + nxt["weights"] + nxt["act_out"] <= sram):
            traffic -= 2 * l["act_out"]
        t_compute, t_memory = cyc / (mhz * 1e6), traffic / (bw * 1e9)
        latency += max(t_compute, t_memory)
        compute_bound += t_compute >= t_memory
        total_cycles += cyc
        total_traffic += traffic
        mac += l["mac"]
    skip = SKIP_MB.get(name, 0) * 1e6
    if skip and sram < skip + max_ws / 2:
        total_traffic += 2 * skip
        latency += 2 * skip / (bw * 1e9)
    return {"model": name, "array": array, "sram_kb": sram_kb, "bw_gbs": bw,
            "utilization": mac / (total_cycles * array * array), "traffic_mb": total_traffic / 1e6,
            "latency_ms": latency * 1e3, "compute_bound_layers": compute_bound, "layers": len(layers)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mhz", type=float, default=500)
    args = parser.parse_args()
    out = ROOT / "compare_results" / "roofline" / "accel_model"
    out.mkdir(exist_ok=True)

    results = []
    for name in MODELS:
        layers = load_layers(name)
        for array in ARRAYS:
            for sram_kb in SRAM_KB:
                for bw in BW_GBS:
                    results.append(evaluate(name, layers, array, sram_kb, bw, args.mhz))
    with open(out / "sweep.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in results)

    def pick(name, array, sram_kb, bw):
        return next(r for r in results if r["model"] == name and r["array"] == array and r["sram_kb"] == sram_kb and r["bw_gbs"] == bw)

    lines = ["# 加速器分析模型(L2)", "", f"由 `accel_model.py` 產生;時脈 {args.mhz:.0f} MHz、INT8、weight-stationary,假設見程式開頭。", ""]
    for name, label in MODELS.items():
        lines += [f"## {label}", "", "PE 利用率(與 SRAM、頻寬無關):", "",
                  "| 陣列 | " + " | ".join(f"{a}×{a}" for a in ARRAYS) + " |", "| --- |" + " --- |" * len(ARRAYS),
                  "| 利用率 | " + " | ".join(f"{pick(name, a, 1024, 12.8)['utilization']:.1%}" for a in ARRAYS) + " |", "",
                  "DRAM 流量 MB(與陣列無關):", "",
                  "| SRAM | " + " | ".join(f"{s // 1024} MB" if s >= 1024 else f"{s} KB" for s in SRAM_KB) + " |",
                  "| --- |" + " --- |" * len(SRAM_KB),
                  "| 流量 | " + " | ".join(f"{pick(name, 32, s, 12.8)['traffic_mb']:.0f}" for s in SRAM_KB) + " |", "",
                  "延遲 ms(DRAM 12.8 GB/s;括號內為運算受限的層數 / 總層數):", "",
                  "| 陣列 \\ SRAM | " + " | ".join(f"{s // 1024} MB" if s >= 1024 else f"{s} KB" for s in SRAM_KB) + " |",
                  "| --- |" + " --- |" * len(SRAM_KB)]
        for a in ARRAYS:
            cells = []
            for s in SRAM_KB:
                r = pick(name, a, s, 12.8)
                cells.append(f"{r['latency_ms']:.1f} ({r['compute_bound_layers']}/{r['layers']})")
            lines.append(f"| {a}×{a} | " + " | ".join(cells) + " |")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"stage2_unet_r34": "#4c72b0", "stage2_yolov8n-seg_fused": "#dd8452"}

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for name, label in MODELS.items():
        ax.plot(SRAM_KB, [pick(name, 32, s, 12.8)["traffic_mb"] for s in SRAM_KB], marker="o", color=colors[name], label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(SRAM_KB, [f"{s // 1024}M" if s >= 1024 else f"{s}K" for s in SRAM_KB])
    ax.set_xlabel("On-chip SRAM (bytes)")
    ax.set_ylabel("DRAM traffic per inference (MB)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "traffic_vs_sram.png", dpi=200)

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for name, label in MODELS.items():
        ax.plot(ARRAYS, [pick(name, a, 1024, 12.8)["utilization"] * 100 for a in ARRAYS], marker="o", color=colors[name], label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ARRAYS, [f"{a}x{a}" for a in ARRAYS])
    ax.set_xlabel("PE array")
    ax.set_ylabel("PE utilization (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "utilization_vs_array.png", dpi=200)

    fig, axes = plt.subplots(1, len(MODELS), figsize=(9.5, 3.6), sharey=False)
    for ax, (name, label) in zip(axes, MODELS.items()):
        for a, style in zip(ARRAYS, ["-", "--", "-.", ":"]):
            ax.plot(SRAM_KB, [pick(name, a, s, 12.8)["latency_ms"] for s in SRAM_KB], style, marker=".", label=f"{a}x{a}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(SRAM_KB, [f"{s // 1024}M" if s >= 1024 else f"{s}K" for s in SRAM_KB])
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("On-chip SRAM")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("Latency per inference (ms)")
    axes[-1].legend(fontsize=8, title="PE array", title_fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(f"INT8, {args.mhz:.0f} MHz, DRAM 12.8 GB/s", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "latency.png", dpi=200)
    print(f"\n輸出: {out}")


if __name__ == "__main__":
    main()
