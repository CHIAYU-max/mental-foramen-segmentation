import argparse
import importlib
import statistics
import subprocess
import time
import types

import cv2
import numpy as np
from ultralytics import YOLO

from compare_yolo_unet import ROOT
from pipeline_timing import TEETH_MODEL, test_panos

# 驗證第一階段 IoU 加速(2026-10-25): 只在兩顆牙外框重疊的區域算交集,聯集用 |A| + |B| - |A ∩ B|。
#   --target fdi_cropped / sort_teeth: 同一份第一階段 YOLO 輸出分別交給 git HEAD 的舊版與目前的新版 process_and_separate_jaws,
#   比對上下顎牙齒(id、cx、cy、y_max,fdi_cropped 另含 FDI 編號)是否完全相同,並比較兩者的時間(不含 YOLO 推論;
#   sort_teeth 的版本會自己讀圖,所以兩邊都多含一次 imread)。
#   另外檢查每顆牙外框內的像素數 = 整張遮罩的像素數(fillPoly 不會畫到外框以外;sort_teeth 的外框本來就由遮罩算出)。
EXTRA_DIR = ROOT / "dental_project" / "20260826__mf_851"


class Cached:
    # 回傳事先算好的 YOLO 結果,讓新舊版本吃到同一份輸入,計時也不含推論
    def __init__(self, results):
        self.results = results

    def __call__(self, *args, **kwargs):
        return self.results


def load_old(target):
    src = subprocess.run(["git", "show", f"HEAD:{target}.py"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    module = types.ModuleType(f"{target}_old")
    exec(compile(src, f"{target}_old.py", "exec"), module.__dict__)
    return module


def key(teeth):
    return [(t["id"], t["cx"], t["cy"], t["y_max"], t.get("fdi_id")) for t in teeth]


def run(target, module, img, path, model):
    start = time.perf_counter()
    if target == "sort_teeth":
        _, upper, lower = module.process_and_separate_jaws(str(path), model)
    else:
        upper, lower = module.process_and_separate_jaws(img, model)
    ms = (time.perf_counter() - start) * 1e3
    if target == "fdi_cropped":
        lower = module.assign_fdi_indices(lower)
    return key(upper), key(lower), ms


def bbox_covers_mask(results, shape):
    H, W = shape[:2]
    for poly in results[0].masks.xy:
        if len(poly) == 0:
            continue
        pts = poly.astype(np.int32)
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        bx, by, bw, bh = cv2.boundingRect(pts)
        if np.count_nonzero(mask[max(by, 0):min(by + bh, H), max(bx, 0):min(bx + bw, W)]) != np.count_nonzero(mask):
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["fdi_cropped", "sort_teeth"], default="fdi_cropped")
    parser.add_argument("--extra", type=int, default=100, help=f"另外再取 {EXTRA_DIR.name} 的前 N 張全景片")
    args = parser.parse_args()
    panos = test_panos() + sorted(EXTRA_DIR.glob("*.jpg"))[:args.extra]
    old = load_old(args.target)
    new = importlib.import_module(args.target)
    yolo = YOLO(TEETH_MODEL)
    mismatches, bbox_bad, old_ms, new_ms, teeth = [], [], [], [], []
    for i, path in enumerate(panos, 1):
        img = cv2.imread(str(path))
        results = yolo(img, retina_masks=True, verbose=False)
        if not results or not results[0].masks:
            continue
        model = Cached(results)
        o_upper, o_lower, o_ms = run(args.target, old, img, path, model)
        n_upper, n_lower, n_ms = run(args.target, new, img, path, model)
        old_ms.append(o_ms)
        new_ms.append(n_ms)
        teeth.append(len(results[0].masks.xy))
        if (o_upper, o_lower) != (n_upper, n_lower):
            mismatches.append(path.name)
        if not bbox_covers_mask(results, img.shape):
            bbox_bad.append(path.name)
        if i % 20 == 0:
            print(f"[{i}/{len(panos)}] 舊 {statistics.median(old_ms):.1f} ms / 新 {statistics.median(new_ms):.1f} ms")
    print(f"\n全景片 {len(old_ms)} 張,每張 YOLO 候選中位數 {statistics.median(teeth)} 個")
    print(f"process_and_separate_jaws(不含推論)中位數: 舊 {statistics.median(old_ms):.1f} ms、新 {statistics.median(new_ms):.1f} ms;"
          f"平均: 舊 {statistics.mean(old_ms):.1f} ms、新 {statistics.mean(new_ms):.1f} ms(快 {statistics.mean(old_ms) / statistics.mean(new_ms):.1f} 倍)")
    print(f"[{args.target}] 上下顎牙齒與 FDI 編號不一致: {mismatches or '無'}")
    print(f"外框外仍有遮罩像素: {bbox_bad or '無'}")


if __name__ == "__main__":
    main()
