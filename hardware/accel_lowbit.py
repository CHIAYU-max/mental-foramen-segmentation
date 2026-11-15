import csv
import math

from hardware.accel_explore import (ARRAYS, BW_GBS, E_DRAM, ROOT, SRAM_KB, e_sram, evaluate, load_policies, skip_bytes)
from hardware.accel_model import load_layers

# 加速器分析模型 L2.5 加位元寬度與 2:4 結構化稀疏(2026-11-10,CPU,使用者要求;見 EXPERIMENTS.md「硬體: 結構化稀疏與 4-bit 的評估與規劃」)。
#   沿用 accel_explore.py 的 weight-stationary 週期、切塊、層間保留與 skip 規則,只把每個元素的 bytes 換成位元寬度 / 8;
#   INT8 dense 的結果必須與 accel_explore.evaluate 完全相同(main 開頭逐組檢查)。仍是一階解析模型,不是 RTL。
#   假設(都是樂觀的上限,沒有外部核對):
#     1. 同樣的 PE 數與時脈;每 PE 每週期 1 個(稀疏時是 1 個非零權重的)MAC。面積不在模型內: 4 位元 PE 較小,同面積可以放更多 PE,這裡不算。
#     2. 2:4 稀疏(NVIDIA Ampere 的做法): 每 4 個權重留 2 個,存值 + 每個 2 位元索引;陣列每列一次收 2 個活化值、用索引選 1 個,
#        所以 K 方向的週期減半(ceil(K/2)),活化值的讀取量不變,權重 bytes = 一半 × (位元寬度 + 2) / 8。索引選擇電路的成本不算。
#     3. MAC 能耗: 乘法 0.2 pJ × (w·a / 64)(乘法器面積約與位元數乘積成正比),加法維持 32 位元累加 0.1 pJ;稀疏時只算實際做的一半。
#     4. 部分和仍是 3 bytes(SRAM 讀寫各一次算 6 bytes),不隨位元寬度變。
#     5. 精度不在這裡: 4 位元與 2:4 的命中率要看 qat_int4.py / 之後的稀疏實驗,這裡的命中率都沿用 INT8 前的數字。
#   輸出: compare_results/roofline/accel_lowbit/(sweep.csv、summary.md)。
CONFIGS = [("INT8", 8, 8, False), ("W4A8", 4, 8, False), ("W4A4", 4, 4, False),
           ("INT8 2:4", 8, 8, True), ("W4A8 2:4", 4, 8, True)]
MHZ = 500
MODEL = "stage2_unet_r34"


def factors(w_bits, a_bits, sparse):
    fw = 0.5 * (w_bits + 2) / 8 if sparse else w_bits / 8   # 每個「原本的」權重平均幾 bytes
    return fw, a_bits / 8


def dram_traffic_lb(l, sram, fw, fa):
    w, ai, ao = l["weights"] * fw, l["act_in"] * fa, l["act_out"] * fa
    if w + ai + ao <= sram:
        return w + ai + ao
    best = math.inf
    tm = 1
    while tm <= l["M"]:
        for tn in sorted({min(l["N"], 2 ** i) for i in range(0, 13)}):
            if tm * l["Cin"] * fa + l["K"] * tn * fw + tm * tn * fa > sram:
                continue
            best = min(best, ai * math.ceil(l["N"] / tn) + w * math.ceil(l["M"] / tm) + ao)
        tm *= 2
    return best


def evaluate_lb(layers, array, sram_kb, bw, w_bits, a_bits, sparse):
    fw, fa = factors(w_bits, a_bits, sparse)
    sram = sram_kb * 1024
    max_ws = max(l["weights"] * fw + l["act_in"] * fa + l["act_out"] * fa for l in layers)
    cycles = traffic_total = latency = mac_done = onchip = 0.0
    dense_mac = 0.0
    for i, l in enumerate(layers):
        k = math.ceil(l["K"] / 2) if sparse else l["K"]
        tiles = math.ceil(k / array) * math.ceil(l["N"] / array)
        cyc = tiles * (l["M"] + 3 * array - 2) - 1
        r, c = min(array, k), min(array, l["N"])
        wslot = (w_bits + 2) / 8 if sparse else w_bits / 8
        onchip += tiles * (r * c * wslot + l["M"] * r * (2 if sparse else 1) * fa + l["M"] * c * 6)
        traffic = dram_traffic_lb(l, sram, fw, fa)
        nxt = layers[i + 1] if i + 1 < len(layers) else None
        if (nxt and nxt["in_hw"] == l["out_hw"] and l["act_out"] <= nxt["act_in"]
                and (l["act_out"] + nxt["act_out"]) * fa + nxt["weights"] * fw <= sram):
            traffic -= 2 * l["act_out"] * fa
        latency += max(cyc / (MHZ * 1e6), traffic / (bw * 1e9))
        cycles += cyc
        traffic_total += traffic
        dense_mac += l["mac"]
        mac_done += l["mac"] / 2 if sparse else l["mac"]
    skip = skip_bytes(layers) * fa
    if sram < skip + max_ws / 2:
        traffic_total += 2 * skip
        latency += 2 * skip / (bw * 1e9)
    e_mac = (0.2e-12 * w_bits * a_bits / 64 + 0.1e-12)
    return {"utilization": mac_done / (cycles * array * array), "traffic_mb": traffic_total / 1e6,
            "latency_ms": latency * 1e3, "e_mac_mj": mac_done * e_mac * 1e3,
            "e_sram_mj": onchip * e_sram(sram_kb) * 1e3, "e_dram_mj": traffic_total * E_DRAM * 1e3}


def main():
    out = ROOT / "compare_results" / "roofline" / "accel_lowbit"
    out.mkdir(exist_ok=True)
    layers = load_layers(MODEL)
    # INT8 dense 必須等於 accel_explore.py(weight-stationary)
    for array in ARRAYS:
        for sram_kb in SRAM_KB:
            for bw in BW_GBS:
                a, b = evaluate(layers, array, sram_kb, bw, MHZ, "ws"), evaluate_lb(layers, array, sram_kb, bw, 8, 8, False)
                for key in a:
                    assert math.isclose(a[key], b[key], rel_tol=1e-9), (array, sram_kb, bw, key, a[key], b[key])
    print(f"INT8 dense 與 accel_explore.py 相同: {len(ARRAYS) * len(SRAM_KB) * len(BW_GBS)} 組")

    policies = [p for p in load_policies() if p[1] == MODEL]
    rows = []
    for label, w, a, sparse in CONFIGS:
        for array in ARRAYS:
            for sram_kb in SRAM_KB:
                for bw in BW_GBS:
                    single = evaluate_lb(layers, array, sram_kb, bw, w, a, sparse)
                    for policy, _, runs, _ in policies:
                        r = {"config": label, "policy": policy, "models_run": round(runs, 3), "array": array,
                             "sram_kb": sram_kb, "bw_gbs": bw, "utilization": single["utilization"]}
                        r.update({k: single[k] * runs for k in ("traffic_mb", "latency_ms", "e_mac_mj", "e_sram_mj", "e_dram_mj")})
                        r["energy_mj"] = r["e_mac_mj"] + r["e_sram_mj"] + r["e_dram_mj"]
                        rows.append(r)
    with open(out / "sweep.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in rows)

    def pick(config, policy, array, sram_kb=1024, bw=12.8):
        return next(r for r in rows if r["config"] == config and r["policy"] == policy and r["array"] == array
                    and r["sram_kb"] == sram_kb and r["bw_gbs"] == bw)

    lines = ["# 加速器分析模型 L2.5:位元寬度與 2:4 稀疏", "",
             f"由 `accel_lowbit.py` 產生;{MHZ} MHz、weight-stationary、每個 ROI;同樣 PE 數,不含面積;假設見程式開頭。"
             "命中率不在這張表(要看 4-bit QAT 與稀疏實驗)。", ""]
    for policy in ("R34/512 five-fold", "R34/512 cascade"):
        for bw in (12.8, 4.0):
            lines += [f"## {policy}(SRAM 1 MB、DRAM {bw} GB/s)", "",
                      "| 設定 | 32×32 ms | 64×64 ms | 64×64 利用率 | DRAM 流量 MB | 32×32 能耗 mJ(MAC / SRAM / DRAM) | 能耗相對 INT8 |",
                      "| --- | --- | --- | --- | --- | --- | --- |"]
            base = pick("INT8", policy, 32, bw=bw)["energy_mj"]
            for label, *_ in CONFIGS:
                s, m = pick(label, policy, 32, bw=bw), pick(label, policy, 64, bw=bw)
                lines.append(f"| {label} | {s['latency_ms']:.0f} | {m['latency_ms']:.0f} | {m['utilization']:.1%} | "
                             f"{s['traffic_mb']:.0f} | {s['energy_mj']:.1f}({s['e_mac_mj']:.1f} / {s['e_sram_mj']:.1f} / "
                             f"{s['e_dram_mj']:.1f}) | {s['energy_mj'] / base:.2f} |")
            lines.append("")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"輸出: {out}")


if __name__ == "__main__":
    main()
