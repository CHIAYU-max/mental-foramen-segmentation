"""2026-11-15 筆電 CPU 速度探測的基準(cpu_probe.ps1 每 10 分鐘呼叫一次;追「同一模型在不同時段差到 1.7 倍」的原因)。
固定工作量、隨機輸入、不需要病人影像: U-Net R34/512 INT8(s0_fold0_int8.xml,1×3×512×512)與第一階段牙齒模型 OpenVINO FP32(1×3×1024×1024)。
各暖機 3 次後量 N 次,印一行 JSON: {"unet_int8_ms": 中位數, "stage1_ms": 中位數, ...}
"""
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov

import deployment.no_throttle as no_throttle

THROTTLE_OFF = no_throttle.disable()   # 2026-11-16: 背景執行時 Windows 節能排程會讓第一階段慢約 1.5 倍

ROOT = Path(__file__).resolve().parents[2]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 15
core = ov.Core()
models = {
    "unet_int8": (ROOT / "compare_results/int8_pano/models/s0_fold0_int8.xml", (1, 1, 512, 512)),   # 灰階單通道(同 unet_int8_bench.py)
    "stage1": (next((ROOT / "dental_project/train_success_replica3/weights/best_openvino_model").glob("*.xml")), (1, 3, 1024, 1024)),
}
out = {}
rng = np.random.default_rng(0)
for name, (xml, shape) in models.items():
    compiled = core.compile_model(str(xml), "CPU", {"PERFORMANCE_HINT": "LATENCY"})
    x = rng.random(shape, dtype=np.float32)
    req = compiled.create_infer_request()
    for _ in range(3):
        req.infer({0: x})
    t = []
    for _ in range(N):
        t0 = time.perf_counter()
        req.infer({0: x})
        t.append((time.perf_counter() - t0) * 1e3)
    out[f"{name}_ms"] = round(statistics.median(t), 2)
    out[f"{name}_p90_ms"] = round(float(np.percentile(t, 90)), 2)
print(json.dumps(out))
