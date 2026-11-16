"""2026-11-15 長時間負載對照(sustained.ps1 呼叫;追「雙螢幕在長時間負載下會不會拖慢 CPU」)。
固定工作量、隨機輸入、不需要病人影像: 每一圈 = 第一階段牙齒模型 OpenVINO FP32(1×3×1024×1024)1 次 + U-Net INT8(1×1×512×512)5 次,
和正式部署流程每張全景片的模型呼叫數差不多(5.3 次 U-Net)。連續跑 --seconds 秒,每次呼叫寫一列: 經過秒數、模型、ms。
用法: python -m deployment.cpu_timing.sustained_bench <輸出 csv> [--seconds 180] [--mode any|pin|pcore|ecore]
11/15 加 --mode(雙螢幕排除後,追「同一設定不同次開跑差 ±25%、頻率不變、兩個模型快慢會對調」是否為 Windows 把執行緒排到 E 核心):
  any   = 現行預設(OpenVINO LATENCY;i5-13500H 上第一階段 FP32 開 12 個執行緒 = 4 P + 8 E 核心,U-Net INT8 開 4 個,都不綁核心)
  pcore = SCHEDULING_CORE_TYPE=PCORE_ONLY(兩個模型都 4 個執行緒、只用 P 核心)
  pin   = ENABLE_CPU_PINNING(執行緒數同 any,OpenVINO 把每個執行緒綁在固定核心上)
  ecore = SCHEDULING_CORE_TYPE=ECORE_ONLY(兩個模型都 8 個執行緒、只用 E 核心;診斷用)
  t8    = 第一階段 INFERENCE_NUM_THREADS=8(預設 12;約 4 P + 4 E),U-Net 不變(4)(11/15 新方向之一)
  (先試過用行程親和性限制核心,但 OpenVINO 不看親和性、照樣開 12 個執行緒,會超額,所以改用 OpenVINO 自己的設定)
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import openvino as ov

import deployment.no_throttle as no_throttle

THROTTLE_OFF = no_throttle.disable()   # 2026-11-16: 背景執行時 Windows 節能排程會讓第一階段慢約 1.5 倍

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("out")
parser.add_argument("--seconds", type=float, default=180)
parser.add_argument("--mode", choices=["any", "pin", "pcore", "ecore", "t8"], default="any")
args = parser.parse_args()
cfg = {"PERFORMANCE_HINT": "LATENCY", **{"any": {}, "pin": {"ENABLE_CPU_PINNING": True}, "pcore": {"SCHEDULING_CORE_TYPE": "PCORE_ONLY"},
                                          "ecore": {"SCHEDULING_CORE_TYPE": "ECORE_ONLY"}, "t8": {}}[args.mode]}
cfg1 = {**cfg, **({"INFERENCE_NUM_THREADS": 8} if args.mode == "t8" else {})}   # 只改第一階段
core = ov.Core()
rng = np.random.default_rng(0)
stage1_c = core.compile_model(str(next((ROOT / "dental_project/train_success_replica3/weights/best_openvino_model").glob("*.xml"))), "CPU", cfg1)
unet_c = core.compile_model(str(ROOT / "compare_results/int8_pano/models/s0_fold0_int8.xml"), "CPU", cfg)
threads = f"{stage1_c.get_property('INFERENCE_NUM_THREADS')}/{unet_c.get_property('INFERENCE_NUM_THREADS')}"
stage1, unet = stage1_c.create_infer_request(), unet_c.create_infer_request()
x1 = rng.random((1, 3, 1024, 1024), dtype=np.float32)
x2 = rng.random((1, 1, 512, 512), dtype=np.float32)
stage1.infer({0: x1})
unet.infer({0: x2})   # 各暖機一次(編譯後的第一次),不計
with open(args.out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t_s", "epoch", "model", "ms"])
    t0 = time.perf_counter()
    while (now := time.perf_counter() - t0) < args.seconds:
        for name, req, x in [("stage1", stage1, x1)] + [("unet_int8", unet, x2)] * 5:
            a = time.perf_counter()
            req.infer({0: x})
            b = time.perf_counter()
            w.writerow([round(a - t0, 3), round(time.time(), 3), name, round((b - a) * 1e3, 2)])
print(f"done mode={args.mode} threads={threads}")
