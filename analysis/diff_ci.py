import csv
import re

import cv2
import numpy as np
from scipy.stats import wilcoxon

from stage2.compare_yolo_unet import GT_DIR, IMAGES_DIR, ROOT, index_files, load_gt
from stage2.kfold_patient import OUT_DIR as PATIENT_DIR
from stage2.kfold_patient import PATIENT_MAP, UNET_TAG, YOLO_TAG
from stage2.kfold_unet_variants import fold_paths
from stage2.kfold_yolo_unet import K_FOLDS, SEED
from analysis.paper_metrics import bootstrap, collect
from analysis.seed_replicates import OUT_DIR as SEED_DIR
from analysis.seed_replicates import SEEDS

# 「U-Net − YOLO」差值的 95% 信賴區間:同一張影像上兩個模型的指標相減,再以病人為單位重抽樣。
# 分別計算各隨機種子,以及三個種子的平均(每張影像先對種子取平均,再相減);
# 並依頦孔可辨識性分組。測試集使用 5 折多數決的預測。
METRICS = [("mf_dice", "頦孔 Dice"), ("mf_hit_1R", "頦孔命中率 1R"), ("ian_dice", "神經管 Dice")]
OUT_CSV = PATIENT_DIR / "diff_ci.csv"
SIDE = re.compile(r"_(Left|Right)_roi$")


def source(seed, tag):
    return (PATIENT_DIR, tag) if seed == SEED else (SEED_DIR, f"{tag}_seed{seed}")


def values(rows, key):
    # 沒有頦孔 GT 的影像沒有命中率,記為 nan
    return np.array([r.get(key, np.nan) for r in rows], dtype=float)


def main():
    with open(PATIENT_MAP, newline="") as f:
        patient_of = {r["stem"]: r["patient"] for r in csv.DictReader(f)}
    visibility_csv = ROOT / "visibility_grading" / "visibility.csv"
    visible = {}
    if visibility_csv.exists():
        with open(visibility_csv, newline="") as f:
            visible = {r["stem"]: int(r["mf_visible"]) for r in csv.DictReader(f)}

    cv_samples = []
    for fold in range(K_FOLDS):
        for path in fold_paths(fold, "val", PATIENT_DIR):
            shape = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).shape
            cv_samples.append((fold, path.stem, load_gt(path.stem, shape, path.parent.parent / "labels", {})))
    test_samples = []
    for stem, path in sorted(index_files((ROOT / IMAGES_DIR).resolve()).items()):
        shape = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).shape
        test_samples.append((stem, load_gt(stem, shape, (ROOT / GT_DIR).resolve(), {})))

    datasets = [
        ("cv", "交叉驗證", [s for _, s, _ in cv_samples], [patient_of[s] for _, s, _ in cv_samples]),
        ("test", "測試集(多數決)", [s for s, _ in test_samples], [SIDE.sub("", s) for s, _ in test_samples]),
    ]
    rng = np.random.default_rng(0)
    records = []
    for dataset, label, stems, patients in datasets:
        # {種子: {模型: 逐張結果}}
        per_seed = {}
        for seed in [SEED] + SEEDS:
            rows = {}
            for model, tag in (("unet", UNET_TAG), ("yolo", YOLO_TAG)):
                root, variant = source(seed, tag)
                rows[model] = collect(root, variant, dataset, cv_samples if dataset == "cv" else [], test_samples if dataset == "test" else [], dataset == "test")
            if rows["unet"] is not None and rows["yolo"] is not None:
                per_seed[seed] = rows
        if not per_seed:
            continue
        seed_sets = [(f"seed {s}", [s]) for s in per_seed] + ([("三種子平均", list(per_seed))] if len(per_seed) > 1 else [])
        groups = [("全部", np.ones(len(stems), dtype=bool))]
        if visible:
            v = np.array([visible.get(s, -1) for s in stems])
            groups += [("可辨識", v == 1), ("無法辨識", v == 0)]

        print(f"\n===== {label}:U-Net − YOLO(95% CI 以病人為單位重抽樣) =====")
        print(f"{'種子':<12}{'組別':<8}{'n':>5}" + "".join(f"{name:>36}" for _, name in METRICS))
        for seed_label, seeds in seed_sets:
            for group, mask in groups:
                text = f"{seed_label:<12}{group:<8}{int(mask.sum()):>5}"
                for key, name in METRICS:
                    u = np.nanmean([values(per_seed[s]["unet"], key) for s in seeds], axis=0)
                    y = np.nanmean([values(per_seed[s]["yolo"], key) for s in seeds], axis=0)
                    d = u - y
                    keep = mask & ~np.isnan(d)
                    est, low, high = bootstrap([{"d": x} for x in d[keep]], [p for p, k in zip(patients, keep) if k], "d", np.mean, rng)
                    p = float(wilcoxon(d[keep]).pvalue) if np.any(d[keep] != 0) else 1.0
                    text += f"{est:>+12.4f} ({low:+.4f} ~ {high:+.4f})"
                    records.append({
                        "dataset": label, "seed": seed_label, "group": group, "metric": name, "n": int(keep.sum()),
                        "unet": round(float(np.mean(u[keep])), 4), "yolo": round(float(np.mean(y[keep])), 4),
                        "diff": round(est, 4), "ci_low": round(low, 4), "ci_high": round(high, 4), "wilcoxon_p": p,
                    })
                print(text)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\n命中率的差值以比例表示(0.06 = 6 個百分點)。逐項結果(含 p 值): {OUT_CSV}")


if __name__ == "__main__":
    main()
