import csv
import sys
from multiprocessing import Pool

import cv2
import numpy as np

from stage2.compare_yolo_unet import GT_DIR, IMAGES_DIR, ROOT, index_files, load_gt
from stage2.kfold_pano import EXCLUDE_TEST, PROJECT, build_groups
from stage2.kfold_unet_variants import fold_paths
from stage2.kfold_yolo_unet import K_FOLDS
from stage2.mf_postproc import largest
from analysis.paper_metrics import SIDE, read_masks
from analysis.seed_bootstrap import cluster_test

# 2026-10-30 命中門檻的敏感度: 主結果用「質心誤差 <= 1R」(R = GT 面積換算的等效半徑),這裡確認 U-Net 的優勢不是只出現在 1R。
#   模型: U-Net(r34_imagenet_aug)與 YOLO26s,種子 0–4;頦孔一律 lcc。CV 用各折驗證預測,測試用五折逐像素多數決再 lcc(同主結果)。
#   每張 ROI 算 error / R(沒有預測 = 無限大)。門檻 0.5、1、1.5、2 R 的命中率,以及 error / R 的中位數。
#   配對檢定同 seed_bootstrap.py: 每張先對種子取平均,CV 用 build_groups、測試用病人。1R 應等於主結果(93.0% / 87.7%),當作檢查。
#   輸出 compare_results/kfold_pano_mf_559/hit_threshold_lcc.csv 與 hit_threshold_per_roi_lcc.csv。只讀已存遮罩(CPU)。
OUT_DIR = ROOT / "compare_results" / PROJECT
SEEDS = (0, 1, 2, 3, 4)
MODELS = {"unet": "r34_imagenet_aug", "yolo": "yolo26s-seg"}
THRESHOLDS = (0.5, 1.0, 1.5, 2.0)
DATA = {}


def relative_error(pred, gt):
    if not pred.any():
        return float("inf")
    radius = (gt.sum() / np.pi) ** 0.5
    return float(np.linalg.norm(np.argwhere(pred).mean(0) - np.argwhere(gt).mean(0))) / radius


def run(job):
    name, seed = job
    tag = MODELS[name] + ("" if seed == 0 else f"_s{seed}")
    rows = []
    for fold, stem, gt in DATA["cv"]:
        rows.append(("cv", stem, name, seed, relative_error(largest(read_masks(OUT_DIR, "cv", tag, fold, stem)["mf"]), gt)))
    for stem, gt in DATA["test"]:
        preds = [read_masks(OUT_DIR, "test", tag, f, stem)["mf"] for f in range(K_FOLDS)]
        rows.append(("test", stem, name, seed, relative_error(largest(np.sum(preds, axis=0) > K_FOLDS / 2), gt)))
    return rows


def main():
    sys.stdout.reconfigure(line_buffering=True)
    for fold in range(K_FOLDS):
        for path in fold_paths(fold, "val", OUT_DIR):
            shape = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).shape
            mf = load_gt(path.stem, shape, path.parent.parent / "labels", {})["mf"]
            if mf.any():
                DATA.setdefault("cv", []).append((fold, path.stem, mf))
    test_gt = (ROOT / GT_DIR).resolve()
    for stem, path in index_files((ROOT / IMAGES_DIR).resolve()).items():
        if stem not in EXCLUDE_TEST:
            shape = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).shape
            mf = load_gt(stem, shape, test_gt, {})["mf"]
            if mf.any():
                DATA.setdefault("test", []).append((stem, mf))
    print(f"CV {len(DATA['cv'])} 張、測試 {len(DATA['test'])} 張(只算有 GT 頦孔的)")

    with Pool(10) as pool:
        rows = [r for part in pool.map(run, [(n, s) for n in MODELS for s in SEEDS]) for r in part]
    with open(OUT_DIR / "hit_threshold_per_roi_lcc.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["setting", "stem", "model", "seed", "error_over_R"])
        writer.writerows(rows)

    errors = {}
    for setting, stem, name, seed, value in rows:
        errors.setdefault(setting, {}).setdefault(stem, {}).setdefault(name, {})[seed] = value
    groups, _, _, _ = build_groups(list(errors["cv"]))
    cluster_of = {"cv": {s: i for i, members in enumerate(groups) for s in members},
                  "test": {s: SIDE.sub("", s) for s in errors["test"]}}
    rng = np.random.default_rng(0)
    results = []
    for setting in ("cv", "test"):
        stems = sorted(errors[setting])
        for threshold in THRESHOLDS:
            rate = {n: np.array([[errors[setting][s][n][k] <= threshold for k in SEEDS] for s in stems], dtype=float) for n in MODELS}
            clusters = {}
            for index, stem in enumerate(stems):
                d = float(rate["unet"][index].mean() - rate["yolo"][index].mean())
                total, n = clusters.get(cluster_of[setting][stem], (0.0, 0))
                clusters[cluster_of[setting][stem]] = (total + d, n + 1)
            est, low, high, p = cluster_test(list(clusters.values()), rng)
            results.append({"setting": setting, "threshold_R": threshold, "n": len(stems), "unet": round(float(rate["unet"].mean()), 4), "yolo26s": round(float(rate["yolo"].mean()), 4),
                            "diff": round(est, 4), "ci_low": round(low, 4), "ci_high": round(high, 4), "p": round(p, 4)})
        for name in MODELS:
            median = np.median([[errors[setting][s][name][k] for k in SEEDS] for s in stems])
            print(f"{setting} {name}: error / R 中位數 {median:.3f}")
    with open(OUT_DIR / "hit_threshold_lcc.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
