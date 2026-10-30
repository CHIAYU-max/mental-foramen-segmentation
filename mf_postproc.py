import argparse
import csv
import math
import statistics
import sys
from multiprocessing import Pool

import cv2
import numpy as np

from compare_yolo_unet import GT_DIR, IMAGES_DIR, ROOT, index_files, load_gt
from kfold_pano import EXCLUDE_TEST, PROJECT
from kfold_unet_variants import fold_paths
from kfold_yolo_unet import K_FOLDS
from paper_metrics import pixel_metrics, read_masks

# 頦孔後處理(2026-10-28): 預測的頦孔只留面積最大的連通區塊(lcc)。
#   kfold_pano.evaluate 沒有做任何後處理,但一個 ROI 只有一個頦孔;多出的小塊會把質心拉離 GT(例如 365_Right_roi)。
#   mf_position_eval.py 已經只取最大區塊,這裡看主表的指標(mf Dice、命中 1R)會怎麼變。
#   規則事先固定(不調參數),只讀已存的遮罩(CPU)。CV / 測試單一模型: 每個模型的遮罩各自處理;
#   測試多數決: 先逐像素多數決再處理。ian 不處理。
#   輸出 compare_results/kfold_pano_mf_559/summary_postproc.csv(none 應與 summary*.csv 相同)與 per_roi_postproc.csv。
OUT_DIR = ROOT / "compare_results" / PROJECT
POSTS = ("none", "lcc")
DATA = {}


def largest(mask):
    if not mask.any():
        return mask
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def apply_post(pred, post):
    # 給其他腳本用的 --post 選項: lcc 時頦孔只留最大區塊,ian 不變
    return {**pred, "mf": largest(pred["mf"])} if post == "lcc" else pred


def mf_metrics(pred, gt):
    # 與 paper_metrics.image_metrics 的 mf_dice、mf_hit_1R、mf_predicted 相同
    row = {"mf_dice": pixel_metrics(pred, gt)["dice"]}
    if gt.any():
        radius = (gt.sum() / np.pi) ** 0.5
        row["mf_predicted"] = float(pred.any())
        error = float(np.linalg.norm(np.argwhere(pred).mean(0) - np.argwhere(gt).mean(0))) if pred.any() else float("nan")
        row["mf_hit_1R"] = float(error <= radius) if error == error else 0.0
    return row


def run(tag):
    per_roi = []
    for fold, stem, gt in DATA["cv"]:
        pred = read_masks(OUT_DIR, "cv", tag, fold, stem)["mf"]
        for post in POSTS:
            per_roi.append({"model": tag, "setting": "cv", "post": post, "stem": stem, **mf_metrics(largest(pred) if post == "lcc" else pred, gt)})
    for stem, gt in DATA["test"]:
        preds = [read_masks(OUT_DIR, "test", tag, f, stem)["mf"] for f in range(K_FOLDS)]
        vote = np.sum(preds, axis=0) > K_FOLDS / 2
        for post in POSTS:
            fix = largest if post == "lcc" else (lambda m: m)
            single = [mf_metrics(fix(p), gt) for p in preds]
            per_roi.append({"model": tag, "setting": "test single", "post": post, "stem": stem,
                            **{k: float(np.nanmean([m[k] for m in single])) for k in single[0]}})
            per_roi.append({"model": tag, "setting": "test ensemble", "post": post, "stem": stem, **mf_metrics(fix(vote), gt)})
    return per_roi


def mcnemar(gained, lost):
    # 精確 McNemar: 不一致的 ROI 做雙尾二項檢定
    n = gained + lost
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(gained, lost) + 1)) / 2 ** n)


def complete(tag):
    return all(len(list((OUT_DIR / "masks" / d / tag / f"fold{f}").glob("*.png"))) > 0 for d in ("cv", "test") for f in range(K_FOLDS))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", default=None, help="masks 底下的模型名稱;不給則用五折都有遮罩的全部模型")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    tags = args.tags or sorted(p.name for p in (OUT_DIR / "masks" / "test").iterdir() if complete(p.name))

    for fold in range(K_FOLDS):
        for path in fold_paths(fold, "val", OUT_DIR):
            shape = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).shape
            DATA.setdefault("cv", []).append((fold, path.stem, load_gt(path.stem, shape, path.parent.parent / "labels", {})["mf"]))
    test_gt = (ROOT / GT_DIR).resolve()
    for stem, path in index_files((ROOT / IMAGES_DIR).resolve()).items():
        if stem not in EXCLUDE_TEST:
            shape = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).shape
            DATA.setdefault("test", []).append((stem, load_gt(stem, shape, test_gt, {})["mf"]))
    print(f"模型 {len(tags)} 個;CV {len(DATA['cv'])} 張、測試 {len(DATA['test'])} 張")

    with Pool(args.workers) as pool:
        per_roi = [r for rows in pool.map(run, tags) for r in rows]

    summary = []
    for tag in tags:
        for setting in ("cv", "test single", "test ensemble"):
            rows = {post: [r for r in per_roi if r["model"] == tag and r["setting"] == setting and r["post"] == post] for post in POSTS}
            base = {r["stem"]: r["mf_hit_1R"] for r in rows["none"]}
            for post in POSTS:
                row = {"model": tag, "setting": setting, "post": post, "n": len(rows[post])}
                for k in ("mf_dice", "mf_hit_1R", "mf_predicted"):
                    values = [r[k] for r in rows[post] if k in r and r[k] == r[k]]
                    row[k] = round(statistics.mean(values), 4)
                if post != "none" and setting != "test single":
                    gained = sum(r["mf_hit_1R"] > base[r["stem"]] for r in rows[post])
                    lost = sum(r["mf_hit_1R"] < base[r["stem"]] for r in rows[post])
                    row.update(hit_gained=gained, hit_lost=lost, mcnemar_p=round(mcnemar(gained, lost), 4))
                summary.append(row)

    fields = ["model", "setting", "post", "n", "mf_dice", "mf_hit_1R", "mf_predicted", "hit_gained", "hit_lost", "mcnemar_p"]
    with open(OUT_DIR / "summary_postproc.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    with open(OUT_DIR / "per_roi_postproc.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "setting", "post", "stem", "mf_dice", "mf_hit_1R", "mf_predicted"])
        writer.writeheader()
        writer.writerows(per_roi)
    print(f"\n{'model':<28}{'setting':<16}{'post':<6}{'mf_dice':>9}{'hit_1R':>9}{'得/失':>8}{'p':>8}")
    for r in summary:
        extra = f"{r['hit_gained']:>4}/{r['hit_lost']:<3}{r['mcnemar_p']:>8.3f}" if "hit_gained" in r else ""
        print(f"{r['model']:<28}{r['setting']:<16}{r['post']:<6}{r['mf_dice']:>9.4f}{r['mf_hit_1R']:>9.4f}  {extra}")
    print(f"\n彙總結果: {OUT_DIR / 'summary_postproc.csv'}")


if __name__ == "__main__":
    main()
