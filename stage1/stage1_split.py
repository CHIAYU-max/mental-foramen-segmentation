"""2026-11-13 第一階段在筆電 CPU 上的時間拆分(模型推論 vs ultralytics 後處理 vs 專案自己的後處理),Tufts ×1.58 影像,不需要病人影像。
輸出: compare_results/deploy_eval/stage1_split_<主機>/per_image.csv
"""
import csv
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from stage2.compare_yolo_unet import ROOT
from stage1.fdi_cropped import assign_fdi_indices, process_and_separate_jaws
from deployment.pipeline_timing import TEETH_MODEL

folder = Path(sys.argv[1])
sizes = [int(v) for v in sys.argv[2:]] or [1024, 768]
paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".png"))[:50]
yolo = YOLO(TEETH_MODEL.parent / f"{TEETH_MODEL.stem}_openvino_model", task="segment")


class Rec:
    def __init__(self, imgsz):
        self.imgsz, self.speed, self.call = imgsz, None, 0.0

    def __call__(self, *a, **k):
        t0 = time.perf_counter()
        r = yolo(*a, imgsz=self.imgsz, device="intel:cpu", **k)
        self.call = (time.perf_counter() - t0) * 1000
        self.speed = r[0].speed
        return r


rows = []
for rnd in range(3):
    for size in sizes:
        rec = Rec(size)
        for i, p in enumerate(paths):
            img = cv2.imread(str(p))
            t0 = time.perf_counter()
            _, lower = process_and_separate_jaws(img, rec)
            t1 = time.perf_counter()
            lower = assign_fdi_indices(lower) if lower else []
            t2 = time.perf_counter()
            if rnd == 0 and i < 3:
                continue
            sp = rec.speed
            rows.append({"round": rnd, "imgsz": size, "image": p.name, "h": img.shape[0], "w": img.shape[1], "teeth": len(lower),
                         "pre": sp["preprocess"], "infer": sp["inference"], "post": sp["postprocess"], "yolo_call": rec.call,
                         "separate_jaws_rest": (t1 - t0) * 1000 - rec.call, "assign_fdi": (t2 - t1) * 1000, "total": (t2 - t0) * 1000})
out = ROOT / "compare_results" / "deploy_eval" / f"stage1_split_{platform.node()}"
out.mkdir(parents=True, exist_ok=True)
with open(out / "per_image.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for size in sizes:
    r = [x for x in rows if x["imgsz"] == size]
    print(size, "n", len(r), {k: round(float(np.mean([x[k] for x in r])), 1) for k in ("pre", "infer", "post", "yolo_call", "separate_jaws_rest", "assign_fdi", "total")})
