"""Accelerator analysis model for the deployed model, with skips and residuals tracked per layer (2026-11-22, CPU).

accel_lowbit.py (L2.5 with bit widths and N:M sparsity) was run on the main model (R34/512, 40.9 G MACs) with
accel_explore's all-or-nothing skip rule. Since 2026-11-18 the deployed stage 2 is the slim decoder
(r34_imagenet_aug_slimhalf, 24.1 G MACs) in the INT8 cascade, 2.58 models per ROI on average
(compare_results/deploy_eval/score_slimhalf/summary.md). This script:
  1. traces both models' layers with skip_traffic.trace (shapes only, random weights) and checks that the main
     model's trace equals compare_results/roofline/workload/stage2_unet_r34_layers.csv;
  2. writes the slim model's layer table next to it (stage2_unet_r34_slimhalf_layers.csv);
  3. evaluates every accel_lowbit configuration two ways:
       old     = accel_lowbit.evaluate_lb unchanged (checked to reproduce accel_lowbit on the main model);
       tracked = the same per-layer cycles, tiling, on-chip energy and next-layer rule, but each skip and residual
                 identity either stays on chip for its lifetime (less SRAM for the layers in between) or costs
                 2 x its bytes in DRAM traffic and time (skip_traffic.py's model, greedy smallest-first, kept only
                 if it lowers the traffic); bytes scale with the activation width.
All the accel_lowbit assumptions hold (optimistic upper bounds, no area, no external check).
Outputs: compare_results/roofline/accel_deploy/(sweep.csv, summary.md).
--tiling v2 (2026-11-22): per-layer DRAM traffic from accel_lowbit.dram_traffic_lb2 (tiling fixed after the Timeloop
cross-check) instead of dram_traffic_lb, for the tracked rule; old rule unchanged; outputs in accel_deploy/v2/.
"""
import argparse
import csv
import math

from hardware.accel_explore import E_DRAM, ROOT, e_sram
from hardware.accel_lowbit import CONFIGS, MHZ, dram_traffic_lb, dram_traffic_lb2, evaluate_lb, factors
from hardware.accel_model import load_layers
from hardware.skip_traffic import trace

OUT = ROOT / "compare_results" / "roofline" / "accel_deploy"
WORKLOAD = ROOT / "compare_results" / "roofline" / "workload"
POLICIES = [("main R34/512 five-fold", "r34_imagenet_aug", 5.0),
            ("slim deployed cascade", "r34_imagenet_aug_slimhalf", 2.58),
            ("slim single model", "r34_imagenet_aug_slimhalf", 1.0)]
ARRAYS = [32, 64]
SRAM_KB = [256, 512, 1024, 2048, 4096]
BW_GBS = [4.0, 12.8]
TILING = dram_traffic_lb


def evaluate_tracked(layers, lives, kept, array, sram_kb, bw, w_bits, a_bits, sparse):
    fw, fa = factors(w_bits, a_bits, sparse)
    sram = sram_kb * 1024
    held = [0.0] * len(layers)
    for lv in lives:
        if lv["name"] in kept:
            for j in lv["layers"]:
                held[j] += lv["bytes"] * fa
    cycles = traffic_total = latency = mac_done = onchip = 0.0
    for i, l in enumerate(layers):
        avail = sram - held[i]
        if avail <= 0:
            return None
        k = math.ceil(l["K"] * sparse / 4) if sparse else l["K"]
        tiles = math.ceil(k / array) * math.ceil(l["N"] / array)
        cyc = tiles * (l["M"] + 3 * array - 2) - 1
        r, c = min(array, k), min(array, l["N"])
        wslot = (w_bits + 2) / 8 if sparse else w_bits / 8
        onchip += tiles * (r * c * wslot + l["M"] * r * (4 / sparse if sparse else 1) * fa + l["M"] * c * 6)
        traffic = TILING(l, avail, fw, fa)
        if traffic == math.inf:
            return None
        nxt = layers[i + 1] if i + 1 < len(layers) else None
        if (nxt and nxt["in_hw"] == l["out_hw"] and l["act_out"] <= nxt["act_in"]
                and (l["act_out"] + nxt["act_out"]) * fa + nxt["weights"] * fw <= sram - max(held[i], held[i + 1])):
            traffic -= 2 * l["act_out"] * fa
        latency += max(cyc / (MHZ * 1e6), traffic / (bw * 1e9))
        cycles += cyc
        traffic_total += traffic
        mac_done += l["mac"] * sparse / 4 if sparse else l["mac"]
    spill = sum(2 * lv["bytes"] * fa for lv in lives if lv["name"] not in kept)
    traffic_total += spill
    latency += spill / (bw * 1e9)
    e_mac = 0.2e-12 * w_bits * a_bits / 64 + 0.1e-12
    return {"utilization": mac_done / (cycles * array * array), "traffic_mb": traffic_total / 1e6,
            "spill_mb": spill / 1e6, "latency_ms": latency * 1e3, "e_mac_mj": mac_done * e_mac * 1e3,
            "e_sram_mj": onchip * e_sram(sram_kb) * 1e3, "e_dram_mj": traffic_total * E_DRAM * 1e3}


def plan_tracked(layers, lives, *args):
    kept = set()
    best = evaluate_tracked(layers, lives, kept, *args)
    for lv in sorted(lives, key=lambda lv: lv["bytes"]):
        r = evaluate_tracked(layers, lives, kept | {lv["name"]}, *args)
        if r and (best is None or r["traffic_mb"] < best["traffic_mb"]):
            kept.add(lv["name"])
            best = r
    best["kept_skips"] = sum(1 for lv in lives if lv["kind"] == "skip" and lv["name"] in kept)
    return best


def main():
    global OUT, TILING
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiling", choices=["v1", "v2"], default="v1")
    args = ap.parse_args()
    if args.tiling == "v2":
        OUT, TILING = OUT / "v2", dram_traffic_lb2
    OUT.mkdir(parents=True, exist_ok=True)
    traced = {}
    for cfg in ("r34_imagenet_aug", "r34_imagenet_aug_slimhalf"):
        traced[cfg] = trace(cfg)
    ref = load_layers("stage2_unet_r34")
    main_layers = traced["r34_imagenet_aug"][0]
    assert len(ref) == len(main_layers)
    for a, b in zip(ref, main_layers):
        for key in ("name", "M", "N", "K", "Cin", "mac", "weights", "act_in", "act_out", "in_hw", "out_hw"):
            assert a[key] == b[key], (a["name"], key, a[key], b[key])
    print(f"main-model trace equals stage2_unet_r34_layers.csv ({len(ref)} layers)")
    for a in (32, 64):
        for kb in (512, 1024):
            for bw in (4.0, 12.8):
                for _, w, act, sp in CONFIGS:
                    x, y = evaluate_lb(ref, a, kb, bw, w, act, sp), evaluate_lb(main_layers, a, kb, bw, w, act, sp)
                    assert all(math.isclose(x[k], y[k], rel_tol=1e-12) for k in x)
    print("old rule on the traced layers reproduces accel_lowbit")

    slim_layers = traced["r34_imagenet_aug_slimhalf"][0]
    with open(WORKLOAD / "stage2_unet_r34_slimhalf_layers.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq", "name", "type", "in_shape", "out_shape", "mac", "weights", "act_in", "act_out", "ws_int8", "oi_int8"])
        for i, l in enumerate(slim_layers):
            kk = l["K"] // l["Cin"]
            kind = {9: "Conv3x3", 1: "Conv1x1", 49: "Conv7x7"}.get(kk, "Conv")
            ws = l["weights"] + l["act_in"] + l["act_out"]
            w.writerow([i, l["name"], kind, f"1x{l['Cin']}x{l['in_hw'][0]}x{l['in_hw'][1]}",
                        f"1x{l['N']}x{l['out_hw'][0]}x{l['out_hw'][1]}", float(l["mac"]), l["weights"], l["act_in"],
                        l["act_out"], ws, l["mac"] / ws])

    rows = []
    for policy, cfg, runs in POLICIES:
        layers, lives = traced[cfg]
        for label, wb, ab, sp in CONFIGS:
            for a in ARRAYS:
                for kb in SRAM_KB:
                    for bw in BW_GBS:
                        for rule in ("old", "tracked"):
                            s = (evaluate_lb(layers, a, kb, bw, wb, ab, sp) if rule == "old"
                                 else plan_tracked(layers, lives, a, kb, bw, wb, ab, sp))
                            r = {"policy": policy, "config": label, "rule": rule, "array": a, "sram_kb": kb, "bw_gbs": bw,
                                 "models_run": runs, "utilization": s["utilization"], "kept_skips": s.get("kept_skips", "")}
                            for k in ("traffic_mb", "latency_ms", "e_mac_mj", "e_sram_mj", "e_dram_mj"):
                                r[k] = s[k] * runs
                            r["energy_mj"] = r["e_mac_mj"] + r["e_sram_mj"] + r["e_dram_mj"]
                            rows.append(r)
    with open(OUT / "sweep.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in rows)

    def pick(policy, config, rule, array, kb=1024, bw=12.8):
        return next(r for r in rows if (r["policy"], r["config"], r["rule"], r["array"], r["sram_kb"], r["bw_gbs"])
                    == (policy, config, rule, array, kb, bw))

    lines = ["# 加速器分析模型：部署版本（瘦身解碼器、串接式）與逐層追蹤 skip / residual", "",
             f"由 `accel_deploy.py` 產生；{MHZ} MHz、weight-stationary、每個 ROI（乘上平均跑的模型數）；"
             "假設同 `accel_lowbit.py`（樂觀上限、不含面積、沒有外部核對）。old = 原本的 skip 規則，tracked = 逐層追蹤。"
             + ("tracked 的逐層 DRAM 流量用 v2 切塊（`dram_traffic_lb2`，Timeloop 核對後修正）；old 不變。" if args.tiling == "v2" else ""), ""]
    for bw in (12.8, 4.0):
        lines += [f"## SRAM 1 MB、DRAM {bw} GB/s", "",
                  "| 策略 | 設定 | 64×64 ms old → tracked | 32×32 ms tracked | DRAM MB old → tracked | 32×32 能耗 mJ tracked | 能耗相對主模型 INT8 old |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        base = pick("main R34/512 five-fold", "INT8", "old", 32, bw=bw)["energy_mj"]
        for policy, *_ in POLICIES:
            for label in ("INT8", "INT8 2:4", "INT8 1:4", "W4A8"):
                o, t = pick(policy, label, "old", 64, bw=bw), pick(policy, label, "tracked", 64, bw=bw)
                t32 = pick(policy, label, "tracked", 32, bw=bw)
                lines.append(f"| {policy} | {label} | {o['latency_ms']:.0f} → {t['latency_ms']:.0f} | {t32['latency_ms']:.0f} | "
                             f"{o['traffic_mb']:.0f} → {t['traffic_mb']:.0f} | {t32['energy_mj']:.1f} | {t32['energy_mj'] / base:.2f} |")
        lines.append("")
    lines += ["## 部署版本（slim deployed cascade）INT8 + 2:4，64×64，tracked：SRAM 大小的影響", "",
              "| SRAM | 12.8 GB/s ms | 4 GB/s ms | DRAM MB | 留住的 skip |", "| ---: | ---: | ---: | ---: | ---: |"]
    for kb in SRAM_KB:
        h, l4 = pick("slim deployed cascade", "INT8 2:4", "tracked", 64, kb, 12.8), pick("slim deployed cascade", "INT8 2:4", "tracked", 64, kb, 4.0)
        lines.append(f"| {kb} KB | {h['latency_ms']:.0f} | {l4['latency_ms']:.0f} | {h['traffic_mb']:.0f} | {h['kept_skips']} / 4 |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
