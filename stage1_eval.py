import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import binomtest
from ultralytics import YOLO

from compare_yolo_unet import ROOT
from fdi_cropped import assign_fdi_indices, find_mandibular_border, get_split_roi_bounds, process_and_separate_jaws
from kfold_pano import ROI_SOURCES
from pipeline_timing import PANO_DIR, TEETH_MODEL

# 第一階段(全景片 → 左右 ROI)的前處理與推論尺寸比較(2026-10-25),不需要牙齒標註、也不重新訓練。
#   目前伺服器上的 ROI(20260826__mf_851/output/roi_cropped)與原始全景片逐像素相同,第一階段吃的是原圖;
#   但牙齒模型訓練時的輸入有做 CLAHE(使用者確認),推論與訓練的分布不一致;CLAHE 也沒有像第二階段那樣系統比較過。
#   這裡把全景片先做不同前處理(clahe2 = 與訓練一致),再以不同 imgsz 跑同一個牙齒模型
#   與 fdi_cropped.py 的整套裁切(牙齒過濾、FDI、找下顎骨下緣也都在前處理後的影像上做,與實際接上 CLAHE 時相同)。
# GT: ROI 的 mf 標註,用 roi_sources.csv 記錄的 ROI 位置換回全景片座標;只用比對可信的 ROI
#   (分數 ≥ 0.9、比第二名高 ≥ 0.06,與 locate_roi_sources.py 相同)。
# 指標(以每個 GT 頦孔為單位):
#   covered    mf 多邊形的外框完全落在某個自動 ROI 內(主要指標;沒裁出 ROI 算沒涵蓋)
#   margin_px  mf 外框到該 ROI 邊界的最短距離,負值 = 超出
#   ref_iou    自動 ROI 與當初 ROI 位置的最大 IoU(只供參考: 當初的 ROI 多半也是原圖跑同一流程裁的,會偏向 raw)
# 與基準(raw、imgsz 1024 = 牙齒模型的訓練尺寸)逐個頦孔成對比較,以 McNemar 精確檢定看涵蓋率差異。
#   注意: 同一位病人的左右側不獨立,p 值只供參考;牙齒模型的訓練影像(alc_66)不在伺服器上,無法確認是否與這些全景片重疊。
TRAIN_LABELS = ROOT / "dental_project" / "20260909" / "mf_559_original" / "labels"
TEST_LABELS = ROOT / "runs" / "gt" / "labels"
OUT_DIR = ROOT / "compare_results" / "stage1_eval"
MF_CLASS = 0
MIN_SCORE, MIN_GAP = 0.9, 0.06
# 名稱: (clipLimit, tileGridSize);None = 原圖。clahe2 即 teeth_preprocess.py 的設定
PREPROCS = {
    "raw": None,
    "clahe1": (1.0, 8),
    "clahe2": (2.0, 8),
    "clahe3": (3.0, 8),
    "clahe4": (4.0, 8),
    "clahe2_t16": (2.0, 16),
}
BASELINE = ("raw", 1024)


class Sized:
    # process_and_separate_jaws 呼叫 model(img, retina_masks=True, verbose=False),這裡補上 imgsz
    def __init__(self, model, imgsz):
        self.model = model
        self.imgsz = imgsz

    def __call__(self, *args, **kwargs):
        return self.model(*args, imgsz=self.imgsz, **kwargs)


def preprocess(img, name):
    if PREPROCS[name] is None:
        return img
    clip, tile = PREPROCS[name]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(gray), cv2.COLOR_GRAY2BGR)


def load_gt():
    # {來源全景片: [{stem, set, box: mf 外框, ref: 當初的 ROI}]}(全景片座標)
    gt = defaultdict(list)
    with open(ROI_SOURCES, newline="") as f:
        for r in csv.DictReader(f):
            if not r["source_pano"] or float(r["score"]) < MIN_SCORE:
                continue
            if r["second_score"] and float(r["score"]) - float(r["second_score"]) < MIN_GAP:
                continue
            label = (TRAIN_LABELS if r["set"] == "train" else TEST_LABELS) / f"{r['stem']}.txt"
            if not label.exists():
                continue
            x, y, w, h = (int(r[k]) for k in "xywh")
            for line in label.read_text().splitlines():
                values = line.split()
                if not values or int(values[0]) != MF_CLASS:
                    continue
                pts = np.array(values[1:], dtype=float).reshape(-1, 2) * (w, h) + (x, y)
                gt[r["source_pano"]].append({
                    "stem": r["stem"], "set": r["set"],
                    "box": (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()),
                    "ref": (x, y, x + w, y + h),
                })
    return gt


def auto_rois(img, model):
    # 與 fdi_cropped.py 的 __main__ 相同的裁切,只回傳座標 (side, x1, y1, x2, y2)
    _, lower = process_and_separate_jaws(img, model)
    if not lower:
        return []
    lower = assign_fdi_indices(lower)
    h, w = img.shape[:2]
    rois = []
    for side, b in get_split_roi_bounds(lower).items():
        jaw_y = min(find_mandibular_border(img, b["y_max_root"], b["x_min"], b["x_max"], b["dynamic_fallback"]), h)
        x2 = min(b["x_max"], w)
        if jaw_y > b["y_upper"] and x2 > b["x_min"]:
            rois.append((side, b["x_min"], b["y_upper"], x2, jaw_y))
    return rois


def box_iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def score_mf(mf, rois):
    x1, y1, x2, y2 = mf["box"]
    margin, iou = -math.inf, 0.0
    for _, a, b, c, d in rois:
        margin = max(margin, min(x1 - a, y1 - b, c - x2, d - y2))
        iou = max(iou, box_iou(mf["ref"], (a, b, c, d)))
    return {"covered": int(margin >= 0), "margin_px": round(margin, 1) if rois else float("nan"), "ref_iou": round(iou, 4)}


def summarize(rows, conditions, out):
    by_cond = defaultdict(dict)
    for r in rows:
        by_cond[(r["preproc"], r["imgsz"])][(r["pano"], r["stem"], r["mf"])] = r
    base = by_cond[BASELINE]
    summary = []
    for cond in conditions:
        items = by_cond[cond]
        panos = {k[0]: r for k, r in items.items()}
        test = [r for r in items.values() if r["set"] == "test"]
        margins = [r["margin_px"] for r in items.values() if r["margin_px"] == r["margin_px"]]
        worse = sum(1 for k, r in items.items() if base[k]["covered"] and not r["covered"])
        better = sum(1 for k, r in items.items() if not base[k]["covered"] and r["covered"])
        summary.append({
            "preproc": cond[0], "imgsz": cond[1], "panos": len(panos), "mf": len(items),
            "pano_2_rois": round(statistics.mean(r["n_rois"] == 2 for r in panos.values()), 4),
            "pano_no_roi": round(statistics.mean(r["n_rois"] == 0 for r in panos.values()), 4),
            "mf_covered": round(statistics.mean(r["covered"] for r in items.values()), 4),
            "mf_covered_test": round(statistics.mean(r["covered"] for r in test), 4) if test else float("nan"),
            "median_margin_px": round(statistics.median(margins), 1) if margins else float("nan"),
            "median_ref_iou": round(statistics.median(r["ref_iou"] for r in items.values()), 4),
            "vs_base_better": better, "vs_base_worse": worse,
            "mcnemar_p": round(binomtest(min(better, worse), better + worse, 0.5).pvalue, 4) if better + worse else 1.0,
        })
    with open(out / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"\n基準: {BASELINE[0]} imgsz {BASELINE[1]};better / worse = 相對基準多涵蓋 / 少涵蓋的頦孔數")
    print(" | ".join(summary[0]))
    for s in summary:
        print(" | ".join(str(v) for v in s.values()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocs", nargs="+", default=list(PREPROCS), choices=list(PREPROCS))
    parser.add_argument("--imgsz", type=int, nargs="+", default=[768, 1024, 1280])
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 張全景片(試跑用)")
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    conditions = [(p, s) for p in args.preprocs for s in args.imgsz]
    if BASELINE not in conditions:
        conditions.insert(0, BASELINE)

    gt = load_gt()
    panos = sorted(gt)[:args.limit or None]
    print(f"全景片 {len(panos)} 張、GT 頦孔 {sum(len(gt[p]) for p in panos)} 個;條件 {len(conditions)} 種")
    yolo = YOLO(TEETH_MODEL)
    rows = []
    for i, pano in enumerate(panos, 1):
        raw = cv2.imread(str(PANO_DIR / pano))
        for name in dict.fromkeys(p for p, _ in conditions):
            img = preprocess(raw, name)
            for preproc, imgsz in conditions:
                if preproc != name:
                    continue
                rois = auto_rois(img, Sized(yolo, imgsz))
                for j, mf in enumerate(gt[pano]):
                    rows.append({"preproc": preproc, "imgsz": imgsz, "pano": pano, "stem": mf["stem"], "mf": j, "set": mf["set"], "n_rois": len(rois), **score_mf(mf, rois)})
        if i % 20 == 0 or i == len(panos):
            print(f"[{i}/{len(panos)}] {pano}")
    with open(out / "per_mf.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summarize(rows, conditions, out)


if __name__ == "__main__":
    main()
