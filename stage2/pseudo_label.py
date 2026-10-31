import argparse
import csv
import os
import sys

import cv2
import numpy as np

from stage2.compare_yolo_unet import NAMES, ROOT
from stage2.kfold_pano import PROJECT
from stage2.kfold_yolo_unet import K_FOLDS
from stage2.mf_postproc import largest
from analysis.pano_uncertainty import disagreement
from analysis.paper_metrics import read_masks
from analysis.unlabeled_pool import OUT_IMAGES
from analysis.unlabeled_shift import OUT, TAGS

# 半監督訓練的假標註(2026-10-28,CPU): 第 k 折的學生只用第 k 折的老師(同一份訓練資料,5 個種子)產生的假標註,
#   老師沒看過第 k 折的驗證集,所以 CV 評分不會因此洩漏;測試集完全不受影響。
#   假標註 = 5 個種子的逐像素多數決,頦孔只留最大區塊(lcc);只保留「多數決有頦孔、而且分歧度 u ≤ 門檻」的 ROI。
#   門檻事先固定為 CV 的 u 第 --keep 分位數(預設 0.8: CV 保留 80% 時命中率最高的那一段),不看測試集。
#   輸出 datasets/20261028/pseudo_fold<k>/{images(連結到 unlabeled_mf851),labels(YOLO 多邊形)}、
#   compare_results/unlabeled/pseudo_fold<k>.txt(該折使用的影像清單,給 kfold_pseudo.py)。
DATA = ROOT / "datasets" / "20261028"


def polygons(mask):
    # 遮罩 → 正規化的多邊形(YOLO 格式);面積太小的碎塊不寫
    h, w = mask.shape
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in contours:
        if cv2.contourArea(c) < 4:
            continue
        c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
        if len(c) >= 3:
            out.append((c / (w, h)).ravel())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", type=float, default=0.8, help="門檻 = CV 的 u 第幾分位數")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    out_dir = ROOT / "compare_results" / PROJECT
    with open(OUT / "shift_per_roi.csv", newline="") as f:
        cv_u = [float(r["u"]) for r in csv.DictReader(f) if r["group"] == "cv"]
    thr = float(np.quantile(cv_u, args.keep))
    print(f"u 門檻 {thr:.4f}(CV 第 {args.keep:.0%} 分位數)")

    paths = sorted(OUT_IMAGES.glob("*.jpg"))
    for fold in range(K_FOLDS):
        root = DATA / f"pseudo_fold{fold}"
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "labels").mkdir(parents=True, exist_ok=True)
        kept = []
        for path in paths:
            vote, u = disagreement([read_masks(out_dir, "unlabeled", t, fold, path.stem) for t in TAGS])
            vote["mf"] = largest(vote["mf"])
            if not vote["mf"].any() or u > thr:
                continue
            lines = [f"{c} " + " ".join(f"{v:.6f}" for v in poly) for c, cls in NAMES.items() for poly in polygons(vote[cls])]
            (root / "labels" / f"{path.stem}.txt").write_text("\n".join(lines) + "\n")
            link = root / "images" / path.name
            if not link.exists():
                os.symlink(path, link)
            kept.append(str(link))
        (OUT / f"pseudo_fold{fold}.txt").write_text("\n".join(kept) + "\n")
        print(f"fold {fold}: 保留 {len(kept)} / {len(paths)} 張")


if __name__ == "__main__":
    main()
