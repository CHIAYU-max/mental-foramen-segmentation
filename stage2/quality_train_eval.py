"""2026-11-05 訓練端方法的評估(CPU,讀已存遮罩;輸出 compare_results/quality_errors/train/summary.md)

比較 kfold_pano 的 --unet-config r34_imagenet_aug_quality / r34_imagenet_aug_hard2(種子 0)與主結果 r34_imagenet_aug 種子 0。
CV = 該折模型、測試 = 五折多數決,都做 lcc;配對差以 folds.csv 的組 bootstrap 2000 次(測試以來源全景片編號為組)。
分層: 第二批、可辨識 = 0、頦孔 CNR 最低三分之一(quality_errors/stage2_per_roi.csv,用 GT 算,只做分層)。
事先定的判準: CV 命中差的 95% CI 下界 > 0 才算有改善;測試只確認方向。
"""
import csv
import sys

import cv2
import numpy as np

from stage2.compare_yolo_unet import GT_DIR, IMAGES_DIR, ROOT, index_files, load_gt
from stage2.kfold_pano import EXCLUDE_TEST, PROJECT
from stage2.kfold_unet_variants import fold_paths
from stage2.kfold_yolo_unet import K_FOLDS
from stage2.mf_postproc import largest
from analysis.paper_metrics import read_masks
from analysis.pano_visibility import VISIBILITY_CSV
from stage2.quality_methods import group_boot, score

MASKS = ROOT / "compare_results" / PROJECT
OUT = ROOT / "compare_results" / "quality_errors" / "train"
BASE = "r34_imagenet_aug"
TAGS = sys.argv[1:] or ["r34_imagenet_aug_quality", "r34_imagenet_aug_hard2"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    folds = {r["stem"]: r for r in csv.DictReader(open(MASKS / "folds.csv", newline=""))}
    visible = {r["stem"]: r["mf_visible"] for r in csv.DictReader(open(VISIBILITY_CSV, newline=""))}
    q = {(r["setting"], r["stem"]): float(r["cnr"]) for r in csv.DictReader(open(ROOT / "compare_results/quality_errors/stage2_per_roi.csv", newline=""))}
    rows = []
    for fold in range(K_FOLDS):
        for path in fold_paths(fold, "val", MASKS):
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            gt = load_gt(path.stem, img.shape, path.parent.parent / "labels", {})["mf"]
            if not gt.any():
                continue
            row = {"setting": "cv", "stem": path.stem, "group": folds[path.stem]["group"], "batch": path.stem.split("_")[0],
                   "visible": visible.get(path.stem, ""), "cnr": q.get(("cv", path.stem), np.nan)}
            for tag in [BASE] + TAGS:
                row[tag + "|hit"], row[tag + "|dice"], row[tag + "|type"] = score(largest(read_masks(MASKS, "cv", tag, fold, path.stem)["mf"]), gt)
            rows.append(row)
    test_gt = (ROOT / GT_DIR).resolve()
    for stem, path in sorted(index_files((ROOT / IMAGES_DIR).resolve()).items()):
        if stem in EXCLUDE_TEST:
            continue
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        gt = load_gt(stem, img.shape, test_gt, {})["mf"]
        if not gt.any():
            continue
        row = {"setting": "test", "stem": stem, "group": stem.split("_")[0], "batch": "test",
               "visible": visible.get(stem, ""), "cnr": q.get(("test", stem), np.nan)}
        for tag in [BASE] + TAGS:
            vote = np.sum([read_masks(MASKS, "test", tag, f, stem)["mf"] for f in range(K_FOLDS)], axis=0) > K_FOLDS / 2
            row[tag + "|hit"], row[tag + "|dice"], row[tag + "|type"] = score(largest(vote), gt)
        rows.append(row)
    with open(OUT / "per_roi.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lines = ["# 頦孔定位: 訓練端的方法(quality_train_eval.py,種子 0)\n"]
    for setting in ("cv", "test"):
        sub = [r for r in rows if r["setting"] == setting]
        cnr = np.array([r["cnr"] for r in sub])
        cut = np.nanpercentile(cnr, 33.3)
        strata = {"全部": sub, "第二批": [r for r in sub if r["batch"] == "second"], "無法辨識": [r for r in sub if r["visible"] == "0"],
                  "CNR 最低 1/3": [r for r in sub if r["cnr"] <= cut], "其餘": [r for r in sub if not r["cnr"] <= cut]}
        lines.append(f"\n## {setting}({len(sub)} 個)\n")
        lines.append("| 設定 | 分層 | n | 命中 | 差(− 主結果)[95% CI] | p | mf Dice 差 | 沒命中 empty / near / far |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for tag in [BASE] + TAGS:
            for name, s in strata.items():
                if not s:
                    continue
                h = np.array([r[tag + "|hit"] for r in s]); h0 = np.array([r[BASE + "|hit"] for r in s])
                d = np.array([r[tag + "|dice"] for r in s]); d0 = np.array([r[BASE + "|dice"] for r in s])
                if tag == BASE:
                    ci = "— | —"
                else:
                    lo, hi, p = group_boot(h - h0, [r["group"] for r in s])
                    ci = f"{100 * (h - h0).mean():+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}] | {p:.3f}"
                t = [r[tag + "|type"] for r in s]
                lines.append(f"| {tag} | {name} | {len(s)} | {100 * h.mean():.2f}% | {ci} | {d.mean() - d0.mean():+.4f} | "
                             f"{t.count('empty')} / {t.count('near')} / {t.count('far')} |")
    text = "\n".join(lines)
    (OUT / "summary.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
