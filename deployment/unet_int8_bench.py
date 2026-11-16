"""2026-11-13 U-Net R34/512 在 CPU 上 OpenVINO INT8(PTQ)比 FP32 快多少(使用者要求;隨機輸入,不需要病人影像)。

- 模型: 種子 0、第 0 折。FP32 = .pth 直接 ov.convert_model;INT8 = int8_pano/models/s0_fold0_int8.xml(dense、PTQ,沒有 QAT、沒有 2:4)。
- 量法: batch 1、輸入 1×1×512×512,同步推論(compiled(x),與部署相同);先各暖機 --warmup 次,
  之後 FP32 / INT8 交替 --rounds 輪、每輪 --iters 次,抵消筆電溫度與背景負載的漂移。報每次推論的中位數、p90,與倍數。
- 另報 OpenVINO 實際執行的精度(FP32 模型在沒有 AMX 的 CPU 上應為 f32)。
- 輸出: compare_results/deploy_eval/unet_bench_<主機>/(per_iter.csv、summary.md)。
"""
import deployment.no_throttle as no_throttle  # 2026-11-16: 背景執行時 Windows 節能排程會讓第一階段慢約 1.5 倍
no_throttle.disable()
import argparse
import csv
import platform
import statistics
import time

import numpy as np
import openvino as ov
import torch

from stage2.compare_yolo_unet import ROOT
from stage2.kfold_pano import PROJECT
from stage2.kfold_patient import UNET_TAG
from stage2.kfold_unet_variants import CONFIGS, build_model

SIZE = CONFIGS[UNET_TAG]["size"]
OUT = ROOT / "compare_results" / "deploy_eval"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    out = OUT / f"unet_bench_{platform.node()}"
    out.mkdir(parents=True, exist_ok=True)
    core = ov.Core()
    config = {"INFERENCE_NUM_THREADS": args.threads} if args.threads else {}
    unet = build_model({**CONFIGS[UNET_TAG], "pretrained": False})
    unet.load_state_dict(torch.load(ROOT / "compare_results" / PROJECT / "weights" / f"{UNET_TAG}_fold0.pth", map_location="cpu")["model_state_dict"])
    models = {"fp32": core.compile_model(ov.convert_model(unet.eval(), example_input=torch.zeros(1, 1, *SIZE)), "CPU", config),
              "int8": core.compile_model(str(ROOT / "compare_results/int8_pano/models/s0_fold0_int8.xml"), "CPU", config)}
    x = (np.random.default_rng(0).integers(0, 256, (1, 1, *SIZE)).astype(np.float32) / 255)
    for m in models.values():
        for _ in range(args.warmup):
            m(x)
    rows = []
    for r in range(args.rounds):
        for name, m in models.items() if r % 2 == 0 else reversed(list(models.items())):
            for i in range(args.iters):
                start = time.perf_counter()
                m(x)
                rows.append({"round": r, "precision": name, "iter": i, "ms": (time.perf_counter() - start) * 1e3})
    with open(out / "per_iter.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    med = {n: statistics.median(r["ms"] for r in rows if r["precision"] == n) for n in models}
    p90 = {n: float(np.percentile([r["ms"] for r in rows if r["precision"] == n], 90)) for n in models}
    per_round = [statistics.median(r["ms"] for r in rows if r["precision"] == "fp32" and r["round"] == k) /
                 statistics.median(r["ms"] for r in rows if r["precision"] == "int8" and r["round"] == k) for k in range(args.rounds)]
    exec_prec = {n: m.get_property("INFERENCE_PRECISION_HINT") for n, m in models.items()}
    lines = [f"# U-Net R34/512 CPU 推論: INT8 vs FP32({platform.node()},{core.get_property('CPU', 'FULL_DEVICE_NAME')},OpenVINO {ov.__version__})", "",
             f"- batch 1、512×512、隨機輸入;暖機各 {args.warmup} 次,交替 {args.rounds} 輪 × {args.iters} 次;執行緒 {args.threads or '預設'};"
             f"INFERENCE_PRECISION_HINT {exec_prec}", "",
             "| 精度 | 中位數 ms | p90 ms |", "| --- | --- | --- |"]
    lines += [f"| {n.upper()} | {med[n]:.1f} | {p90[n]:.1f} |" for n in models]
    lines += ["", f"- INT8 比 FP32 快 {med['fp32'] / med['int8']:.2f} 倍(各輪 {' / '.join(f'{v:.2f}' for v in per_round)})"]
    text = "\n".join(lines)
    (out / "summary.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
