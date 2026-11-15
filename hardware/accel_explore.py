import argparse
import csv
import math
import re
from pathlib import Path

from hardware.accel_model import dram_traffic, load_layers

ROOT = Path(__file__).resolve().parents[1]

# 加速器分析模型 L2.5(2026-10-31):把 L2 從「單一 U-Net」延伸到實際的部署策略,並加上能耗與 output-stationary。
#   沿用 accel_model.py 的切塊與 DRAM 流量(假設 3–6);DRAM 流量當成與資料流無關。不是 RTL,數字只用來比較趨勢。
#   週期(accel_validate.py 用 SCALE-Sim 3.0 核對;比 L2 多算填滿與排空的 R − 2 週期):
#     weight-stationary: ceil(K/R) · ceil(N/C) · (M + 2R + C − 2) − 1
#     output-stationary: ceil(M/R) · ceil(N/C) · (K + R + C − 2) − 1
#   SRAM 存取(INT8;部分和 3 bytes,每次讀寫各一次,所以每個部分和算 6 bytes;R'、C' = 實際用到的列與行):
#     weight-stationary: 每塊 載入權重 R'·C' + 輸入 M·R' + 部分和 M·C'·6
#     output-stationary: 每塊 K·(輸入 R' + 權重 C') + 輸出 R'·C'·3
#   每次推論的能耗(一階,45 nm,Horowitz ISSCC 2014;只比相對大小,不含漏電、互連、控制與 PE 內暫存器):
#     MAC 0.3 pJ(8 位元乘法 0.2 + 32 位元加法 0.1);SRAM 12.5·√(容量 MB) pJ/byte(1 MB 100 pJ/64 位元,單一巨集);
#     DRAM 160 pJ/byte。
#   部署策略的運算量 = 單一模型 × 平均跑的模型數,各模型依序執行(權重每次都從 DRAM 載入)。命中率都是測試集、5 種子、lcc。
#   輸出: compare_results/roofline/accel_explore/(sweep.csv、summary.md、energy_vs_hit.png)。
MODELS = {"stage2_unet_r34": 512, "stage2_unet_r34_384": 384, "stage2_unet_r18_384": 384}
ARRAYS = [8, 16, 32, 48, 64, 96]
SRAM_KB = [128, 256, 512, 1024, 2048, 4096, 8192]
BW_GBS = [4.0, 12.8, 25.6]
DATAFLOWS = ["ws", "os"]
TARGETS_MS = [100, 200, 500]
E_MAC, E_DRAM = 0.3e-12, 160e-12
BASE = ROOT / "compare_results" / "kfold_pano_mf_559"


def e_sram(sram_kb):
    return 12.5e-12 * math.sqrt(sram_kb / 1024)


def skip_bytes(layers):
    last = {}
    for l in layers:
        m = re.match(r"(stem|layers\.\d)", l["name"])
        if m:
            last[m.group(1)] = l
    return sum(last[k]["act_out"] for k in ("stem", "layers.0", "layers.1", "layers.2") if k in last)


def compute_cycles(l, array, df):
    if df == "ws":
        return math.ceil(l["K"] / array) * math.ceil(l["N"] / array) * (l["M"] + 3 * array - 2) - 1
    return math.ceil(l["M"] / array) * math.ceil(l["N"] / array) * (l["K"] + 2 * array - 2) - 1


def sram_bytes(l, array, df):
    r, c = min(array, l["K"] if df == "ws" else l["M"]), min(array, l["N"])
    if df == "ws":
        return math.ceil(l["K"] / array) * math.ceil(l["N"] / array) * (r * c + l["M"] * r + l["M"] * c * 6)
    return math.ceil(l["M"] / array) * math.ceil(l["N"] / array) * (l["K"] * (r + c) + r * c * 3)


def evaluate(layers, array, sram_kb, bw, mhz, df):
    sram = sram_kb * 1024
    max_ws = max(l["weights"] + l["act_in"] + l["act_out"] for l in layers)
    cycles = traffic_total = latency = mac = onchip = 0.0
    for i, l in enumerate(layers):
        cyc = compute_cycles(l, array, df)
        traffic = dram_traffic(l, sram)
        nxt = layers[i + 1] if i + 1 < len(layers) else None
        if (nxt and nxt["in_hw"] == l["out_hw"] and l["act_out"] <= nxt["act_in"]
                and l["act_out"] + nxt["weights"] + nxt["act_out"] <= sram):
            traffic -= 2 * l["act_out"]
        latency += max(cyc / (mhz * 1e6), traffic / (bw * 1e9))
        cycles += cyc
        traffic_total += traffic
        mac += l["mac"]
        onchip += sram_bytes(l, array, df)
    skip = skip_bytes(layers)
    if sram < skip + max_ws / 2:
        traffic_total += 2 * skip
        latency += 2 * skip / (bw * 1e9)
    return {"utilization": mac / (cycles * array * array), "traffic_mb": traffic_total / 1e6, "latency_ms": latency * 1e3,
            "e_mac_mj": mac * E_MAC * 1e3, "e_sram_mj": onchip * e_sram(sram_kb) * 1e3,
            "e_dram_mj": traffic_total * E_DRAM * 1e3}


def load_policies():
    with open(BASE / "cascade_s01234_lcc.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["model"] == "r34_imagenet_aug"]

    def mean(policy, tau, key):
        v = [float(r[key]) for r in rows if r["policy"] == policy and r["tau"] == tau]
        return sum(v) / len(v)

    full = mean("full", "", "hit")
    with open(BASE / "seed_bootstrap_r34_imagenet_aug_384_s01234_lcc.csv", newline="") as f:
        gap = next(float(r["diff"]) for r in csv.DictReader(f) if r["setting"] == "test" and r["metric"] == "mf_hit_1R")
    with open(BASE / "distill" / "single_r18_imagenet_aug_384_lcc_summary.csv", newline="") as f:
        small = next(float(r["hit"]) for r in csv.DictReader(f) if r["row"] == "r18_imagenet_aug_384 single lcc")
    return [("R34/512 five-fold", "stage2_unet_r34", 5.0, full),
            ("R34/512 cascade", "stage2_unet_r34", mean("cascade2", "0.5", "folds_run"), mean("cascade2", "0.5", "hit")),
            ("R34/512 single fold", "stage2_unet_r34", 1.0, mean("single", "", "hit")),
            ("R34/384 five-fold", "stage2_unet_r34_384", 5.0, full - gap),
            ("R18/384 single", "stage2_unet_r18_384", 1.0, small)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mhz", type=float, default=500)
    args = parser.parse_args()
    out = ROOT / "compare_results" / "roofline" / "accel_explore"
    out.mkdir(exist_ok=True)

    policies = load_policies()
    results = []
    for name in MODELS:
        layers = load_layers(name)
        for array in ARRAYS:
            for sram_kb in SRAM_KB:
                for bw in BW_GBS:
                    for df in DATAFLOWS:
                        single = evaluate(layers, array, sram_kb, bw, args.mhz, df)
                        for label, model, runs, hit in policies:
                            if model != name:
                                continue
                            r = {"policy": label, "models_run": round(runs, 3), "test_hit": round(hit, 4), "array": array,
                                 "sram_kb": sram_kb, "bw_gbs": bw, "dataflow": df, "utilization": single["utilization"]}
                            r.update({k: single[k] * runs for k in ("traffic_mb", "latency_ms", "e_mac_mj", "e_sram_mj", "e_dram_mj")})
                            r["energy_mj"] = r["e_mac_mj"] + r["e_sram_mj"] + r["e_dram_mj"]
                            results.append(r)
    with open(out / "sweep.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in results)

    def pick(label, array=32, sram_kb=1024, bw=12.8, df="ws"):
        return next(r for r in results if r["policy"] == label and r["array"] == array and r["sram_kb"] == sram_kb
                    and r["bw_gbs"] == bw and r["dataflow"] == df)

    labels = [p[0] for p in policies]
    lines = ["# 加速器分析模型 L2.5", "", f"由 `accel_explore.py` 產生;{args.mhz:.0f} MHz、INT8,每個 ROI 的數字;能耗只比相對大小,假設見程式開頭。", "",
             "## 部署策略(weight-stationary、SRAM 1 MB、DRAM 12.8 GB/s)", "",
             "| 策略 | 模型數 | 測試命中 | 32×32 ms | 64×64 ms | 32×32 能耗 mJ(MAC / SRAM / DRAM) |", "| --- | --- | --- | --- | --- | --- |"]
    for l in labels:
        a, b = pick(l), pick(l, array=64)
        lines.append(f"| {l} | {a['models_run']:.2f} | {a['test_hit']:.1%} | {a['latency_ms']:.0f} | {b['latency_ms']:.0f} | "
                     f"{a['energy_mj']:.1f}({a['e_mac_mj']:.1f} / {a['e_sram_mj']:.1f} / {a['e_dram_mj']:.1f}) |")
    lines += ["", "## 達到延遲目標的最小陣列(SRAM 1 MB、12.8 GB/s、weight-stationary)", "",
              "| 策略 | " + " | ".join(f"{t} ms" for t in TARGETS_MS) + " |", "| --- |" + " --- |" * len(TARGETS_MS)]
    for l in labels:
        cells = []
        for t in TARGETS_MS:
            hit = next((a for a in ARRAYS if pick(l, array=a)["latency_ms"] <= t), None)
            cells.append(f"{hit}×{hit}" if hit else f"> {ARRAYS[-1]}×{ARRAYS[-1]}")
        lines.append(f"| {l} | " + " | ".join(cells) + " |")
    lines += ["", "## 資料流(R34/512 五折,SRAM 1 MB、12.8 GB/s)", "",
              "| 陣列 | WS 利用率 | OS 利用率 | WS ms | OS ms | WS SRAM mJ | OS SRAM mJ | WS 總能耗 mJ | OS 總能耗 mJ |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for a in ARRAYS:
        w, o = pick(labels[0], array=a), pick(labels[0], array=a, df="os")
        lines.append(f"| {a}×{a} | {w['utilization']:.1%} | {o['utilization']:.1%} | {w['latency_ms']:.0f} | {o['latency_ms']:.0f} | "
                     f"{w['e_sram_mj']:.1f} | {o['e_sram_mj']:.1f} | {w['energy_mj']:.1f} | {o['energy_mj']:.1f} |")
    lines += ["", "## SRAM 對能耗(R34/512 五折,32×32、12.8 GB/s、weight-stationary)", "",
              "| SRAM | " + " | ".join(f"{s // 1024} MB" if s >= 1024 else f"{s} KB" for s in SRAM_KB) + " |",
              "| --- |" + " --- |" * len(SRAM_KB),
              "| SRAM mJ | " + " | ".join(f"{pick(labels[0], sram_kb=s)['e_sram_mj']:.1f}" for s in SRAM_KB) + " |",
              "| DRAM mJ | " + " | ".join(f"{pick(labels[0], sram_kb=s)['e_dram_mj']:.1f}" for s in SRAM_KB) + " |",
              "| 總能耗 mJ | " + " | ".join(f"{pick(labels[0], sram_kb=s)['energy_mj']:.1f}" for s in SRAM_KB) + " |", ""]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    points = [pick(l) for l in labels]
    ax.scatter([p["energy_mj"] for p in points], [p["test_hit"] * 100 for p in points], s=44, color="#4c72b0", zorder=3)
    for p in points:
        left = p["policy"] == "R34/512 cascade"
        ax.annotate(p["policy"], (p["energy_mj"], p["test_hit"] * 100), xytext=(-6 if left else 6, 5), textcoords="offset points",
                    fontsize=8, ha="right" if left else "left")
    ax.set_xscale("log")
    ax.set_xlabel("Modeled energy per ROI (mJ, 32x32, 1 MB SRAM)")
    ax.set_ylabel("Test hit rate (%)")
    ax.grid(alpha=0.3, which="both")
    ax.margins(x=0.25, y=0.15)
    fig.tight_layout()
    fig.savefig(out / "energy_vs_hit.png", dpi=200)
    print(f"\n輸出: {out}")


if __name__ == "__main__":
    main()
