import argparse
import copy
import csv
import pickle
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from stage2.compare_yolo_unet import ROOT
from stage1.fdi_cropped import assign_fdi_indices, get_split_roi_bounds
from stage1.fdi_numbering import assign_fdi_v2

# 比較 FDI 編號舊版(fdi_cropped.assign_fdi_indices)與 v2(fdi_numbering.assign_fdi_v2),2026-10-25。
# 牙齒偵測與過濾都用舊版(process_and_separate_jaws,原圖 imgsz 1024),只換編號,所以差異全部來自編號。
#   第一次執行時把每張全景片的下顎牙齒(外框、質心、面積、輪廓)存成快取,之後調整編號只要用 CPU 重算。
# 沒有 FDI 的 GT,所以看:
#   1. 有 ROI 的側數(舊版失敗、v2 救回的;舊版有、v2 反而沒有的)
#   2. 兩版都有 ROI 時,被編成 x4 / x5 / x6 的是不是同一顆牙(以質心比對)
#   3. 有差異的全景片輸出疊圖(上: 舊版紅字,下: v2 綠字)供目視檢查
PANO_DIR = ROOT / "dental_project" / "20260826__mf_851"
OUT = ROOT / "compare_results" / "fdi_v2"
CACHE = OUT / "lower_teeth.pkl"
SIDES = {"Right": 4, "Left": 3}
OVERLAY_WIDTH = 1200


def build_cache(panos):
    from ultralytics import YOLO
    from stage1.fdi_cropped import process_and_separate_jaws
    from deployment.pipeline_timing import TEETH_MODEL
    from stage1.stage1_eval import Sized
    model = Sized(YOLO(TEETH_MODEL), 1024)
    cache = {}
    for i, path in enumerate(panos, 1):
        img = cv2.imread(str(path))
        _, lower = process_and_separate_jaws(img, model)
        teeth = []
        for t in lower:
            x1, y1, x2, y2 = t["bbox"]
            cs, _ = cv2.findContours(t["mask"][y1:y2, x1:x2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            teeth.append({k: t[k] for k in ("cx", "cy", "bbox", "area", "y_max", "score")} |
                         {"contour": max(cs, key=cv2.contourArea).reshape(-1, 2) + (x1, y1)})
        cache[path.name] = {"shape": img.shape[:2], "lower": teeth}
        if i % 100 == 0:
            print(f"[快取 {i}/{len(panos)}]")
    CACHE.write_bytes(pickle.dumps(cache))
    return cache


def side_info(lower, quadrant):
    # {x4/x5/x6: 質心 x}、該側是否會裁 ROI
    ids = {t["fdi_id"][1]: t["cx"] for t in lower if t.get("quadrant") == quadrant and t.get("fdi_id")}
    return ids


def draw(img, lower, color, label):
    for t in lower:
        cv2.polylines(img, [t["contour"].astype(np.int32)], True, color, 3)
        if t.get("fdi_id"):
            cv2.putText(img, t["fdi_id"], (int(t["cx"]) - 25, int(t["bbox"][1]) - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 4)
    cv2.putText(img, label, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.2, color, 5)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlays", type=int, default=60, help="最多輸出幾張有差異的疊圖")
    parser.add_argument("--gap", type=float, default=None, help="空隙判斷的比例(例如 0.6);不給就不判斷空隙")
    parser.add_argument("--center", type=float, default=0.0, help="中線的影像中央權重")
    args = parser.parse_args()
    out = OUT / f"gap{args.gap}_center{args.center}"
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    panos = sorted(p for p in PANO_DIR.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    cache = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else build_cache(panos)

    rows, changed = [], []
    for name, item in cache.items():
        h, w = item["shape"]
        v1 = assign_fdi_indices(copy.deepcopy(item["lower"])) if item["lower"] else []
        v2 = assign_fdi_v2(copy.deepcopy(item["lower"]), w, args.center, args.gap)
        b1 = set(get_split_roi_bounds(v1)) if v1 else set()
        b2 = set(get_split_roi_bounds(v2)) if v2 else set()
        diff = False
        for side, q in SIDES.items():
            s1, s2 = side_info(v1, q), side_info(v2, q)
            same_456 = all(s1.get(k) == s2.get(k) for k in "456")
            rows.append({"pano": name, "side": side, "lower_teeth": len(item["lower"]),
                         "roi_v1": int(side in b1), "roi_v2": int(side in b2), "same_456": int(same_456),
                         "v1_456": " ".join(f"{k}:{int(s1[k])}" for k in "456" if k in s1),
                         "v2_456": " ".join(f"{k}:{int(s2[k])}" for k in "456" if k in s2)})
            diff |= (side in b1) != (side in b2) or not same_456
        if diff:
            changed.append((name, v1, v2))

    with open(out / "compare.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    n = len(rows)
    print(f"全景片 {len(cache)} 張、{n} 側")
    print(f"有 ROI 的側: 舊版 {sum(r['roi_v1'] for r in rows)},v2 {sum(r['roi_v2'] for r in rows)}")
    print(f"  舊版沒有、v2 有: {sum(1 for r in rows if not r['roi_v1'] and r['roi_v2'])};舊版有、v2 沒有: {sum(1 for r in rows if r['roi_v1'] and not r['roi_v2'])}")
    both = [r for r in rows if r["roi_v1"] and r["roi_v2"]]
    print(f"兩版都有 ROI 的 {len(both)} 側中,x4/x5/x6 對到不同牙齒: {sum(1 for r in both if not r['same_456'])}")
    print(f"有差異的全景片 {len(changed)} 張")
    print("v2 每側最大編號分布:", dict(sorted(Counter(r["v2_456"].count(":") for r in rows).items())))

    for name, v1, v2 in changed[:args.overlays]:
        img = cv2.imread(str(PANO_DIR / name))
        top = draw(img.copy(), v1, (0, 0, 255), "v1")
        bottom = draw(img.copy(), v2, (0, 200, 0), "v2")
        both_img = np.vstack([top, bottom])
        scale = OVERLAY_WIDTH / both_img.shape[1]
        cv2.imwrite(str(out / "overlays" / f"{Path(name).stem}.jpg"), cv2.resize(both_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
