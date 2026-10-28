import argparse
import csv
import re
import statistics
from pathlib import Path

from compare_yolo_unet import ROOT

# 用 roofline 估計模型在醫院常見的入門顯卡上的延遲(2026-10-25)。手邊沒有這些顯卡,只能用模型估計,不是實測。
# 方法:
#   1. 每層延遲 = max(FLOPs / 算力, 搬移量 / 頻寬)(roofline 下限),取自 roofline_profile.py 的 layers.csv。
#   2. 目標顯卡的可達算力 / 頻寬 = 規格值 × RTX 3090 的「實測 / 規格」比例(算力 0.67、頻寬 0.92 左右,從 summary.md 讀)。
#   3. 校正: 同一個模型在 3090 上「實測端到端延遲 / 逐層下限總和」= k(包含 cat、interpolate 等沒有算進逐層的運算與 kernel 啟動),
#      假設在目標顯卡上 k 相同,估計值 = k × 目標顯卡的逐層下限總和。
#   限制: 小顯卡的 kernel 啟動開銷比例、時脈、快取都不同,k 不一定相同;只能當數量級的參考。
# 端到端: pipeline_timing 的 GPU 階段(stage1_inference、stage2_model)依模型的估計 / 3090 實測比例放大,CPU 階段沿用伺服器實測。
ROOFLINE = ROOT / "compare_results" / "roofline"
OUT = ROOT / "compare_results" / "hospital_sim"
SPEC_3090 = (35.6, 936)
# 名稱: (FP32 TFLOP/s, 記憶體頻寬 GB/s, 顯存 GB),官方規格
GPUS = {
    "RTX 3090(伺服器)": (35.6, 936, 24),
    "GTX 1650": (2.98, 128, 4),
    "GTX 1660 Super": (5.03, 336, 6),
    "RTX 3050 8GB": (9.10, 224, 8),
    "RTX A2000 12GB": (7.99, 288, 12),
    "T4": (8.14, 320, 16),
    "RTX 4060": (15.11, 272, 8),
    "RTX 4050 Laptop": (12.1, 192, 6),  # 2560 CUDA cores × 2 × 2.37 GHz(最高加速時脈,實際依筆電功耗而定)
}
# pipeline_timing 實際用的模型 → roofline 資料夾(Ultralytics 推論時會融合 Conv+BN,所以用 _fused)
MODELS = {
    "第一階段 YOLOv8s(牙齒)": "stage1_teeth_yolov8s_fused",
    "U-Net R34": "stage2_unet_r34",
    "YOLOv8n-seg": "stage2_yolov8n-seg_fused",
    "YOLO26s-seg": "stage2_yolo26s-seg_fused",
}
PIPELINES = {"unet": "U-Net R34", "yolov8n-seg": "YOLOv8n-seg"}


def read_summary(d):
    text = (ROOFLINE / d / "summary.md").read_text()
    get = lambda pattern: float(re.search(pattern, text).group(1))
    return get(r"peak compute: ([\d.]+)"), get(r"bandwidth: ([\d.]+)"), get(r"median of 50\): ([\d.]+)")


def bound_ms(d, tflops, gbps):
    with open(ROOFLINE / d / "layers.csv", newline="") as f:
        return sum(max(float(r["flops"]) / (tflops * 1e12), float(r["bytes"]) / (gbps * 1e9)) for r in csv.DictReader(f)) * 1e3


def estimate(d):
    peak, bw, measured = read_summary(d)
    eff_c, eff_b = peak / 1e3 / SPEC_3090[0], bw / SPEC_3090[1]
    k = measured / bound_ms(d, peak / 1e3, bw)
    return measured, {g: k * bound_ms(d, t * eff_c, b * eff_b) for g, (t, b, _) in GPUS.items()}


def main():
    argparse.ArgumentParser().parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    est = {name: estimate(d) for name, d in MODELS.items()}
    lines = ["# 入門顯卡延遲估計(roofline 模型,非實測)", "",
             "單次推論(batch 1)ms;第一列為 RTX 3090 實測,其餘為估計。顯存需求都在 1 GB 以內,4 GB 的顯卡就夠。", "",
             "| 顯卡 | TFLOP/s | GB/s | " + " | ".join(est) + " |", "| --- | --- | --- | " + " | ".join("---" for _ in est) + " |"]
    for g, (t, b, _) in GPUS.items():
        lines.append(f"| {g} | {t} | {b} | " + " | ".join(f"{e[1][g]:.1f}" for e in est.values()) + " |")
    lines += ["", "## 端到端(每張全景片,ms)", "",
              "GPU 階段依上表比例放大,CPU 階段沿用伺服器(i9-10980XE)實測;醫院電腦的 CPU 較慢,CPU 階段請看 CPU 模擬的結果。", ""]
    rows = []
    for tag, name in PIPELINES.items():
        path = ROOFLINE / f"pipeline_{tag}.csv"
        if not path.exists():
            continue
        with open(path, newline="") as f:
            data = list(csv.DictReader(f))
        mean = {c: statistics.mean(float(r[c]) for r in data) for c in data[0] if c not in ("pano", "found")}
        s1, s2 = est["第一階段 YOLOv8s(牙齒)"], est[name]
        for g in GPUS:
            gpu = mean["stage1_inference"] * s1[1][g] / s1[0] + mean["stage2_model"] * s2[1][g] / s2[0]
            cpu = mean["sum_ms"] - mean["stage1_inference"] - mean["stage2_model"]
            rows.append({"pipeline": name, "gpu": g, "gpu_ms": round(gpu, 1), "cpu_ms": round(cpu, 1), "total_ms": round(gpu + cpu, 1)})
    if rows:
        lines += ["| 第二階段 | 顯卡 | GPU 階段 | CPU 階段 | 合計 |", "| --- | --- | --- | --- | --- |"]
        lines += [f"| {r['pipeline']} | {r['gpu']} | {r['gpu_ms']} | {r['cpu_ms']} | {r['total_ms']} |" for r in rows]
        with open(OUT / "gpu_estimate.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (OUT / "gpu_estimate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
