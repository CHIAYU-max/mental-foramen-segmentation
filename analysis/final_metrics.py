import argparse
import csv
import statistics
from pathlib import Path

import cv2
import numpy as np

from stage2.compare_yolo_unet import GT_DIR, IMAGES_DIR, NAMES, ROOT, index_files, load_gt, pixel_metrics
from analysis.diagnose_plateau import tolerant_dice
from stage2.kfold_patient import OUT_DIR, UNET_TAG, YOLO_TAG
from stage2.kfold_unet_variants import fold_paths
from stage2.kfold_yolo_unet import K_FOLDS

# 用 kfold_patient.py 存下的預測遮罩計算論文用的指標,不需要重新預測:
#   - Dice 與容許邊界誤差的 Dice
#   - 頦孔的定位: 中心誤差、誤差在 GT 等效半徑以內的比例、完全沒預測的比例
#   - 測試集另外計算五折模型逐像素多數決(ensemble)的結果
TOLERANCES = [2, 4]
VARIANTS = [YOLO_TAG, UNET_TAG]


def read_masks(dataset, variant, fold, stem):
    # 遮罩以位元旗標儲存: 1 = mf, 2 = ian
    image = cv2.imread(str(OUT_DIR / "masks" / dataset / variant / f"fold{fold}" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SystemExit(f"找不到 {dataset}/{variant}/fold{fold}/{stem}.png,請先執行 kfold_patient.py --eval-only")
    return {cls: (image >> c) & 1 > 0 for c, cls in NAMES.items()}


def image_metrics(pred, gt):
    row = {}
    for cls in NAMES.values():
        row[f"{cls}_dice"] = pixel_metrics(pred[cls], gt[cls])["dice"]
        for t in TOLERANCES:
            row[f"{cls}_tol{t}"] = tolerant_dice(pred[cls], gt[cls], t)
    # 頦孔定位只在 GT 有頦孔的影像上計算;沒有預測時 center 為 nan、hit 為 0
    if gt["mf"].any():
        radius = (gt["mf"].sum() / np.pi) ** 0.5
        row["mf_predicted"] = float(pred["mf"].any())
        row["mf_center"] = float("nan")
        row["mf_hit"] = 0.0
        if pred["mf"].any():
            row["mf_center"] = float(np.linalg.norm(np.argwhere(pred["mf"]).mean(0) - np.argwhere(gt["mf"]).mean(0)))
            row["mf_hit"] = float(row["mf_center"] <= radius)
    return row


def average(rows):
    # 同一張影像在多個模型下的結果取平均(nan 不列入)
    keys = {k for r in rows for k in r}
    out = {}
    for k in keys:
        values = [r[k] for r in rows if k in r and r[k] == r[k]]
        out[k] = statistics.mean(values) if values else float("nan")
    return out


def summarize(label, rows, unit, scale):
    def mean(key):
        values = [r[key] for r in rows if key in r and r[key] == r[key]]
        return statistics.mean(values) if values else float("nan")

    centers = sorted(r["mf_center"] * scale for r in rows if "mf_center" in r and r["mf_center"] == r["mf_center"])
    text = f"{label:<34}{len(rows):>5}"
    for cls in NAMES.values():
        text += f"{mean(f'{cls}_dice'):>9.4f}" + "".join(f"{mean(f'{cls}_tol{t}'):>9.4f}" for t in TOLERANCES)
    text += f"{mean('mf_predicted'):>10.1%}{mean('mf_hit'):>10.1%}{statistics.median(centers):>9.2f}{centers[int(len(centers) * 0.9)]:>9.2f} {unit}"
    print(text)
    return {"setting": label, "n": len(rows), **{k: round(mean(k), 4) for k in sorted({k for r in rows for k in r})}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default=IMAGES_DIR, help="獨立測試集的影像資料夾")
    parser.add_argument("--gt", default=GT_DIR, help="獨立測試集的標註資料夾")
    parser.add_argument("--pixel-mm", type=float, default=None, help="ROI 影像每像素的毫米數;有給的話中心誤差以毫米表示")
    args = parser.parse_args()
    unit, scale = ("mm", args.pixel_mm) if args.pixel_mm else ("px", 1.0)

    header = f"{'設定':<32}{'n':>5}"
    for cls in NAMES.values():
        header += f"{cls + ' dice':>9}" + "".join(f"{f'±{t}px':>9}" for t in TOLERANCES)
    header += f"{'mf 有預測':>7}{'mf 命中':>8}{'誤差中位':>6}{'誤差P90':>8}"
    print("±Npx 為容許 N 像素邊界誤差的 Dice;mf 命中 = 預測中心落在 GT 等效半徑以內;誤差為有預測者的中心距離")
    print(header)

    summary = []
    # 交叉驗證: 每張影像只有一個沒看過它的模型
    for variant in VARIANTS:
        rows = []
        for fold in range(K_FOLDS):
            for path in fold_paths(fold, "val", OUT_DIR):
                shape = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).shape
                gt = load_gt(path.stem, shape, path.parent.parent / "labels", {})
                rows.append(image_metrics(read_masks("cv", variant, fold, path.stem), gt))
        summary.append(summarize(f"cv   {variant}", rows, unit, scale))

    # 測試集: 單一模型的平均表現,以及五個模型多數決後的表現
    test_gt = (ROOT / args.gt).resolve()
    images = index_files((ROOT / args.images).resolve())
    for variant in VARIANTS:
        single, ensemble = [], []
        for stem, path in images.items():
            shape = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).shape
            gt = load_gt(stem, shape, test_gt, {})
            preds = [read_masks("test", variant, fold, stem) for fold in range(K_FOLDS)]
            single.append(average([image_metrics(p, gt) for p in preds]))
            vote = {cls: np.sum([p[cls] for p in preds], axis=0) > K_FOLDS / 2 for cls in NAMES.values()}
            ensemble.append(image_metrics(vote, gt))
        summary.append(summarize(f"test {variant} (單一模型平均)", single, unit, scale))
        summary.append(summarize(f"test {variant} (五折多數決)", ensemble, unit, scale))

    out_csv = OUT_DIR / "final_metrics.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"\n彙總結果: {out_csv}")


if __name__ == "__main__":
    main()
