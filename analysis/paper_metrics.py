import argparse
import csv
import re
import warnings

import cv2
import numpy as np

# 某模型在某張影像上五折都沒有預測時,平均值是 nan,這是預期的情況,不需要警告
warnings.filterwarnings("ignore", message="Mean of empty slice")

from stage2.ablation_patient import ABLATIONS
from stage2.ablation_patient import OUT_DIR as ABLATION_DIR
from stage2.compare_yolo_unet import GT_DIR, IMAGES_DIR, NAMES, ROOT, index_files, load_gt, pixel_metrics
from analysis.diagnose_plateau import tolerant_dice
from stage2.kfold_patient import OUT_DIR as PATIENT_DIR
from stage2.kfold_patient import PATIENT_MAP, UNET_TAG, YOLO_TAG
from stage2.kfold_unet_variants import fold_paths
from stage2.kfold_yolo_unet import K_FOLDS

# 論文用的評估表。沒有可靠的像素間距,而且兩批全景片的解析度不同,
# 所以距離類指標除了像素之外,也以「頦孔 GT 等效半徑 R」為單位報告(與解析度無關)。
#   重疊:     Dice、IoU、容許 2/4 像素邊界誤差的 Dice
#   邊界距離: HD95、ASSD(像素與 R 的倍數)
#   定位:     頦孔有預測的比例、中心誤差落在 0.5R / 1R 以內的比例、中心誤差 / R
# 每個數值附上以「病人」為單位重抽樣的 95% 信賴區間(同一病人的左右側一起抽)。
TOLERANCES = [2, 4]
N_BOOTSTRAP = 2000
SIDE = re.compile(r"_(Left|Right)_roi$")
# 表中的模型: (名稱, 遮罩資料夾, 版本名稱)
MAIN = [("YOLOv8n-seg", PATIENT_DIR, YOLO_TAG), ("U-Net ResNet34 (final)", PATIENT_DIR, UNET_TAG)]
ABLATION_LABELS = {
    "unet_scratch": "U-Net, scratch, basic aug",
    "scratch_aug": "U-Net, scratch, strong aug",
    "r34_scratch_aug": "ResNet34, scratch, strong aug",
    "r34_imagenet": "ResNet34, ImageNet, basic aug",
}


def read_masks(root, dataset, variant, fold, stem):
    image = cv2.imread(str(root / "masks" / dataset / variant / f"fold{fold}" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    return {cls: (image >> c) & 1 > 0 for c, cls in NAMES.items()}


def boundary(mask):
    return mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def surface_distances(pred, gt):
    # 兩個遮罩邊界之間雙向的距離,任一方為空時回傳 None
    if not pred.any() or not gt.any():
        return None
    bp, bg = boundary(pred), boundary(gt)
    to_gt = cv2.distanceTransform((~bg).astype(np.uint8), cv2.DIST_L2, 3)
    to_pred = cv2.distanceTransform((~bp).astype(np.uint8), cv2.DIST_L2, 3)
    return np.concatenate([to_gt[bp], to_pred[bg]])


def image_metrics(pred, gt):
    nan = float("nan")
    row = {}
    radius = (gt["mf"].sum() / np.pi) ** 0.5 if gt["mf"].any() else nan
    for cls in NAMES.values():
        m = pixel_metrics(pred[cls], gt[cls])
        row[f"{cls}_dice"], row[f"{cls}_iou"] = m["dice"], m["iou"]
        for t in TOLERANCES:
            row[f"{cls}_tol{t}"] = tolerant_dice(pred[cls], gt[cls], t)
        d = surface_distances(pred[cls], gt[cls])
        row[f"{cls}_hd95"] = float(np.percentile(d, 95)) if d is not None else nan
        row[f"{cls}_assd"] = float(d.mean()) if d is not None else nan
    row["mf_hd95_R"] = row["mf_hd95"] / radius
    row["mf_assd_R"] = row["mf_assd"] / radius
    if gt["mf"].any():
        row["mf_predicted"] = float(pred["mf"].any())
        error = nan
        if pred["mf"].any():
            error = float(np.linalg.norm(np.argwhere(pred["mf"]).mean(0) - np.argwhere(gt["mf"]).mean(0)))
        row["mf_center_px"] = error
        row["mf_center_R"] = error / radius
        # 沒有預測視為沒命中
        row["mf_hit_0.5R"] = float(error <= 0.5 * radius) if error == error else 0.0
        row["mf_hit_1R"] = float(error <= radius) if error == error else 0.0
    return row


# 平均值類與中位數類的指標;中位數用於分布偏斜的距離
MEAN_KEYS = (
    [f"{c}_{m}" for c in NAMES.values() for m in ["dice", "iou"] + [f"tol{t}" for t in TOLERANCES]]
    + ["mf_predicted", "mf_hit_0.5R", "mf_hit_1R"]
)
MEDIAN_KEYS = ["mf_center_px", "mf_center_R", "mf_hd95", "mf_hd95_R", "mf_assd", "mf_assd_R", "ian_hd95", "ian_assd"]


def bootstrap(rows, patients, key, stat, rng):
    # 以病人為單位重抽樣;回傳 (估計值, 下界, 上界)
    by_patient = {}
    for r, p in zip(rows, patients):
        if key in r and r[key] == r[key]:
            by_patient.setdefault(p, []).append(r[key])
    groups = [np.array(v) for v in by_patient.values()]
    if not groups:
        return float("nan"), float("nan"), float("nan")
    estimate = stat(np.concatenate(groups))
    picks = rng.integers(0, len(groups), (N_BOOTSTRAP, len(groups)))
    if stat is np.mean:
        # 平均值可以用各病人的總和與張數直接算,不必每次串接
        sums = np.array([g.sum() for g in groups])
        counts = np.array([len(g) for g in groups])
        samples = sums[picks].sum(1) / counts[picks].sum(1)
    else:
        samples = [stat(np.concatenate([groups[i] for i in picked])) for picked in picks]
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(estimate), float(low), float(high)


def collect(root, variant, dataset, cv_samples, test_samples, ensemble):
    # 回傳逐張結果;測試集可選擇單一模型平均或五折多數決,缺少遮罩時回傳 None
    rows = []
    if dataset == "cv":
        for fold, stem, gt in cv_samples:
            pred = read_masks(root, "cv", variant, fold, stem)
            if pred is None:
                return None
            rows.append(image_metrics(pred, gt))
        return rows
    for stem, gt in test_samples:
        preds = [read_masks(root, "test", variant, fold, stem) for fold in range(K_FOLDS)]
        if any(p is None for p in preds):
            return None
        if ensemble:
            vote = {cls: np.sum([p[cls] for p in preds], axis=0) > K_FOLDS / 2 for cls in NAMES.values()}
            rows.append(image_metrics(vote, gt))
        else:
            per_model = [image_metrics(p, gt) for p in preds]
            rows.append({k: float(np.nanmean([m.get(k, np.nan) for m in per_model])) for k in per_model[0]})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default=IMAGES_DIR, help="獨立測試集的影像資料夾")
    parser.add_argument("--gt", default=GT_DIR, help="獨立測試集的標註資料夾")
    args = parser.parse_args()
    rng = np.random.default_rng(0)

    with open(PATIENT_MAP, newline="") as f:
        info = {r["stem"]: r for r in csv.DictReader(f)}
    cv_samples, cv_patients, cv_batches = [], [], []
    for fold in range(K_FOLDS):
        for path in fold_paths(fold, "val", PATIENT_DIR):
            shape = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).shape
            cv_samples.append((fold, path.stem, load_gt(path.stem, shape, path.parent.parent / "labels", {})))
            cv_patients.append(info[path.stem]["patient"])
            cv_batches.append(info[path.stem]["batch"])
    test_gt = (ROOT / args.gt).resolve()
    test_samples, test_patients = [], []
    for stem, path in index_files((ROOT / args.images).resolve()).items():
        shape = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).shape
        test_samples.append((stem, load_gt(stem, shape, test_gt, {})))
        test_patients.append(SIDE.sub("", stem))

    models = MAIN + [(ABLATION_LABELS[n], ABLATION_DIR, n) for n in ABLATIONS]
    settings = [("cv", False, "cross-validation"), ("test", False, "test, single model"), ("test", True, "test, 5-fold ensemble")]
    results, by_batch = [], []
    for dataset, ensemble, setting in settings:
        patients = cv_patients if dataset == "cv" else test_patients
        for label, root, variant in models:
            rows = collect(root, variant, dataset, cv_samples, test_samples, ensemble)
            if rows is None:
                print(f"[略過] {label} / {setting}: 遮罩不完整")
                continue
            item = {"setting": setting, "model": label, "n": len(rows)}
            for keys, stat in ((MEAN_KEYS, np.mean), (MEDIAN_KEYS, np.median)):
                for key in keys:
                    est, low, high = bootstrap(rows, patients, key, stat, rng)
                    item[key], item[f"{key}_ci"] = round(est, 4), f"{low:.4f}-{high:.4f}"
            results.append(item)
            if dataset == "cv":
                for batch in sorted(set(cv_batches)):
                    sub = [r for r, b in zip(rows, cv_batches) if b == batch]
                    by_batch.append(
                        (label, batch, len(sub), np.mean([r["mf_dice"] for r in sub]), np.mean([r["mf_hit_1R"] for r in sub if "mf_hit_1R" in r]))
                    )

    def show(title, columns):
        print(f"\n===== {title} =====")
        print(f"{'setting':<24}{'model':<32}{'n':>5}" + "".join(f"{name:>24}" for name, _ in columns))
        for item in results:
            cells = "".join(f"{item[k]:>9.3f} ({item[k + '_ci']:<12})" for _, k in columns)
            print(f"{item['setting']:<24}{item['model']:<32}{item['n']:>5}{cells}")

    show("頦孔:重疊(平均, 95% CI)", [("Dice", "mf_dice"), ("IoU", "mf_iou"), ("Dice ±2px", "mf_tol2"), ("Dice ±4px", "mf_tol4")])
    show(
        "頦孔:定位(比例為平均;距離為中位數, 95% CI)",
        [("predicted", "mf_predicted"), ("hit 0.5R", "mf_hit_0.5R"), ("hit 1R", "mf_hit_1R"), ("center/R", "mf_center_R"), ("HD95/R", "mf_hd95_R")],
    )
    show("頦孔:像素距離(中位數, 95% CI)", [("center px", "mf_center_px"), ("HD95 px", "mf_hd95"), ("ASSD px", "mf_assd")])
    show("下齒槽神經(Dice 等為平均;HD95 為中位數)", [("Dice", "ian_dice"), ("IoU", "ian_iou"), ("Dice ±4px", "ian_tol4"), ("HD95 px", "ian_hd95")])

    print("\n===== 交叉驗證依資料批次分開(兩批全景片的解析度不同) =====")
    for label, batch, n, dice, hit in by_batch:
        print(f"  {label:<32}{batch:<8} n={n:<5} mf Dice {dice:.4f}   hit 1R {hit:.1%}")

    out_csv = PATIENT_DIR / "paper_metrics.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\n彙總結果(含所有指標與信賴區間): {out_csv}")


if __name__ == "__main__":
    main()
