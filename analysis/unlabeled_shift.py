import argparse
import csv
import statistics
import sys

import cv2
import numpy as np
from scipy.stats import ks_2samp

from stage2.compare_yolo_unet import ROOT, load_gt
from stage2.kfold_pano import PROJECT
from stage2.kfold_unet_variants import fold_paths
from stage2.kfold_yolo_unet import K_FOLDS
from stage2.mf_postproc import largest, mf_metrics
from analysis.pano_uncertainty import disagreement
from analysis.pano_visibility import load_test
from analysis.paper_metrics import read_masks
from analysis.unlabeled_pool import OUT_IMAGES

# 沒有標註的 ROI 和有標註的是不是同一種分布(2026-10-28,CPU,不需要標註):
#   有標註的 ROI 可能是比較好標的那些(選樣偏差)。用不需要 GT 的量比較三組:
#     CV(559,每張只有沒看過它的那一折)、測試(137)、沒有標註的池(unlabeled_pool.py,每一折都沒看過)。
#   為了可比較,三組都用「同一折、5 個種子」當成員(與 CV 相同的 5 個成員):
#     有沒有預測頦孔(多數決)、分歧度 u(pano_uncertainty.disagreement)、預測頦孔面積與位置(lcc 後)。
#     測試與沒有標註的池每張有 5 折各一組,視為 5 個樣本(各自都是沒看過的模型)。
#   估計命中率: 依 CV 的 u 分成五等分,各組的命中率(有 GT)乘上另一組落在各區間的比例。先套在測試集上檢查準不準。
#   輸出 compare_results/unlabeled/shift_per_roi.csv、shift_summary.csv。
TAGS = ["r34_imagenet_aug"] + [f"r34_imagenet_aug_s{s}" for s in range(1, 5)]
OUT = ROOT / "compare_results" / "unlabeled"


def features(preds, gt=None):
    vote, u = disagreement(preds)
    mf = largest(vote["mf"])
    h, w = mf.shape
    row = {"mf_present": int(mf.any()), "u": round(u, 4), "mf_area_frac": round(float(mf.mean()), 5)}
    if mf.any():
        y, x = np.argwhere(mf).mean(0)
        row.update(mf_x=round(x / w, 4), mf_y=round(y / h, 4))
    if gt is not None and gt["mf"].any():
        row["hit"] = mf_metrics(mf, gt["mf"])["mf_hit_1R"]
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", default=TAGS)
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    out_dir = ROOT / "compare_results" / PROJECT
    rows = []
    for fold in range(K_FOLDS):
        for path in fold_paths(fold, "val", out_dir):
            gt = load_gt(path.stem, cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).shape, path.parent.parent / "labels", {})
            preds = [read_masks(out_dir, "cv", t, fold, path.stem) for t in args.tags]
            rows.append({"group": "cv", "stem": path.stem, "fold": fold, **features(preds, gt)})
    for stem, gt in load_test():
        for fold in range(K_FOLDS):
            rows.append({"group": "test", "stem": stem, "fold": fold, **features([read_masks(out_dir, "test", t, fold, stem) for t in args.tags], gt)})
    for path in sorted(OUT_IMAGES.glob("*.jpg")):
        for fold in range(K_FOLDS):
            rows.append({"group": "unlabeled", "stem": path.stem, "fold": fold,
                         **features([read_masks(out_dir, "unlabeled", t, fold, path.stem) for t in args.tags])})
    print(f"樣本: " + ", ".join(f"{g} {sum(r['group'] == g for r in rows)}" for g in ("cv", "test", "unlabeled")))

    cv = [r for r in rows if r["group"] == "cv"]
    edges = np.quantile([r["u"] for r in cv], [0.2, 0.4, 0.6, 0.8])
    bin_of = lambda u: int(np.searchsorted(edges, u, side="right"))
    bin_hit = []
    for b in range(5):
        hits = [r["hit"] for r in cv if bin_of(r["u"]) == b and "hit" in r]
        bin_hit.append(statistics.mean(hits))
    print("CV 的 u 五等分邊界:", np.round(edges, 3).tolist(), "各區間命中率:", [round(h, 3) for h in bin_hit])

    summary = []
    base = {k: [r[k] for r in cv] for k in ("u", "mf_area_frac")}
    for group in ("cv", "test", "unlabeled"):
        sub = [r for r in rows if r["group"] == group]
        present = [r for r in sub if r["mf_present"]]
        share = [sum(bin_of(r["u"]) == b for r in sub) / len(sub) for b in range(5)]
        row = {"group": group, "n": len(sub), "mf_present": round(statistics.mean(r["mf_present"] for r in sub), 4),
               "u_median": round(statistics.median(r["u"] for r in sub), 4), "u_p90": round(float(np.quantile([r["u"] for r in sub], 0.9)), 4),
               "u_top20_share": round(share[4], 4), "area_median": round(statistics.median(r["mf_area_frac"] for r in present), 5),
               "x_median": round(statistics.median(r["mf_x"] for r in present), 4), "y_median": round(statistics.median(r["mf_y"] for r in present), 4),
               "hit_estimated": round(sum(s * h for s, h in zip(share, bin_hit)), 4)}
        hits = [r["hit"] for r in sub if "hit" in r]
        row["hit_actual"] = round(statistics.mean(hits), 4) if hits else ""
        if group != "cv":
            row["ks_u_p"] = round(float(ks_2samp([r["u"] for r in sub], base["u"]).pvalue), 4)
            row["ks_area_p"] = round(float(ks_2samp([r["mf_area_frac"] for r in present], [a for a, r in zip(base["mf_area_frac"], cv) if r["mf_present"]]).pvalue), 4)
        summary.append(row)

    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in (("shift_per_roi.csv", rows), ("shift_summary.csv", summary)):
        with open(OUT / name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in data for k in r)))
            writer.writeheader()
            writer.writerows(data)
    for r in summary:
        print(r)
    print(f"\n輸出: {OUT / 'shift_summary.csv'}、shift_per_roi.csv")


if __name__ == "__main__":
    main()
