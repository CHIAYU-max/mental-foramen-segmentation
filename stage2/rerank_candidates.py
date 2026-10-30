import argparse
import csv
import math
import statistics
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from stage2.compare_yolo_unet import GT_DIR, IMAGES_DIR, IOU, NAMES, ROOT, index_files, load_gt, pixel_metrics
from stage2.eval_kfold_variants import FIELDS, MIN_CONF, summarize
from stage2.kfold_yolo_unet import K_FOLDS, OUT_DIR, PROJECT

# 用各折已訓練好的 YOLO 權重,比較「怎麼從候選中挑出 mf 與 ian」的不同做法,不重新訓練
#   yolo_top1:     各類別各自取信心最高的候選(目前的做法)
#   oracle_topK:   從信心前 K 名中取與 GT 最接近的候選,是重新挑選所能達到的上限,不是可用的方法
#   joint_tauT:    mf 與 ian 一起挑,偏好信心高且兩者距離近的組合;T 越小越重視距離
BASELINE = "yolo_top1"
ORACLE_KS = [2, 3, 5, 50]
TAUS = [5, 10, 20, 40]
# 一起挑選時每個類別最多考慮的候選數
JOINT_K = 5


def csv_path(dataset):
    return OUT_DIR / f"rerank_{dataset}.csv"


def get_candidates(result, shape):
    # 回傳 {類別: [(信心, 遮罩)]},依信心由高到低,空遮罩不列入
    candidates = {cls: [] for cls in NAMES.values()}
    if result.masks is not None:
        data = result.masks.data.cpu().numpy() > 0.5
        for m, c, conf in zip(data, result.boxes.cls.int().tolist(), result.boxes.conf.tolist()):
            if m.shape != shape:
                m = cv2.resize(m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            if m.any():
                candidates[NAMES[c]].append((conf, m))
    for items in candidates.values():
        items.sort(key=lambda x: x[0], reverse=True)
    return candidates


def mask_distance(a, b):
    # 兩個遮罩之間的最短距離(像素),重疊或相鄰時為 0
    dist = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 3)
    return float(dist[a].min())


def dice(a, b):
    total = a.sum() + b.sum()
    return 2 * (a & b).sum() / total if total else 1.0


def build_variants(candidates, gt, shape):
    empty = np.zeros(shape, dtype=bool)
    variants = {BASELINE: {cls: items[0][1] if items else empty for cls, items in candidates.items()}}

    for k in ORACLE_KS:
        variants[f"oracle_top{k}"] = {
            cls: max((m for _, m in items[:k]), key=lambda m: dice(m, gt[cls]), default=empty) for cls, items in candidates.items()
        }

    # 只有兩個類別都有候選時才能一起挑,否則沿用各自信心最高的
    mf, ian = candidates["mf"][:JOINT_K], candidates["ian"][:JOINT_K]
    distances = {(i, j): mask_distance(m, n) for i, (_, m) in enumerate(mf) for j, (_, n) in enumerate(ian)}
    for tau in TAUS:
        masks = dict(variants[BASELINE])
        if distances:
            i, j = max(distances, key=lambda ij: math.log(mf[ij[0]][0]) + math.log(ian[ij[1]][0]) - distances[ij] / tau)
            masks = {"mf": mf[i][1], "ian": ian[j][1]}
        variants[f"joint_tau{tau}"] = masks
    return variants


def evaluate(dataset, fold, samples, gt_distances):
    source = OUT_DIR / "yaml" / f"rerank_{dataset}_fold{fold}.txt"
    source.write_text("".join(f"{path}\n" for path, _ in samples))
    gt_dirs = {path.stem: gt_dir for path, gt_dir in samples}

    yolo = YOLO(ROOT / "runs" / "segment" / PROJECT / f"yolo_fold{fold}" / "weights" / "best.pt")
    results = yolo.predict(source=str(source), conf=MIN_CONF, iou=IOU, max_det=50, retina_masks=True, stream=True, verbose=False)
    rows = []
    for result in results:
        stem = Path(result.path).stem
        shape = tuple(result.orig_shape)
        gt = load_gt(stem, shape, gt_dirs[stem], {})
        if fold == 0 and gt["mf"].any() and gt["ian"].any():
            gt_distances.append(mask_distance(gt["mf"], gt["ian"]))

        for name, masks in build_variants(get_candidates(result, shape), gt, shape).items():
            for cls in NAMES.values():
                row = {"dataset": dataset, "fold": fold, "filename": stem, "variant": name, "class": cls}
                rows.append({**row, **pixel_metrics(masks[cls], gt[cls])})
    print(f"[{dataset}] fold {fold}: {len(samples)} 張")
    return rows


def run(dataset, args):
    rows, gt_distances = [], []
    test_images = index_files((ROOT / args.images).resolve())
    test_gt = (ROOT / args.gt).resolve()
    for fold in range(K_FOLDS):
        if dataset == "cv":
            val_txt = OUT_DIR / "yaml" / f"fold{fold}_val.txt"
            paths = [Path(line) for line in val_txt.read_text().splitlines() if line]
            samples = [(p, p.parent.parent / "labels") for p in paths]
        else:
            samples = [(p, test_gt) for p in test_images.values()]
        rows += evaluate(dataset, fold, samples, gt_distances)

    with open(csv_path(dataset), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # 檢查「mf 與 ian 相鄰」這個前提在標註中是否成立
    if gt_distances:
        gt_distances.sort()
        p90 = gt_distances[int(len(gt_distances) * 0.9)]
        print(
            f"[{dataset}] GT 中 mf 與 ian 的最短距離(像素): 中位數 {statistics.median(gt_distances):.1f}, "
            f"第 90 百分位 {p90:.1f}, 最大 {gt_distances[-1]:.1f} (n={len(gt_distances)})"
        )


def per_fold(dataset):
    # 各折分開看目前做法的分數,確認是否有某幾折特別差
    with open(csv_path(dataset), newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["variant"] == BASELINE]
    print(f"\n[{dataset}] {BASELINE} 各折的平均 Dice")
    print(f"{'fold':<6}" + "".join(f"{cls:>10}" for cls in NAMES.values()))
    for fold in sorted({r["fold"] for r in rows}):
        cells = "".join(
            f"{statistics.mean(float(r['dice']) for r in rows if r['fold'] == fold and r['class'] == cls):>10.4f}"
            for cls in NAMES.values()
        )
        print(f"{fold:<6}{cells}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", default=["cv", "test"], choices=["cv", "test"])
    parser.add_argument("--images", default=IMAGES_DIR, help="獨立測試集的影像資料夾")
    parser.add_argument("--gt", default=GT_DIR, help="獨立測試集的標註資料夾")
    parser.add_argument("--summary-only", action="store_true", help="只重新彙總已有結果,不重新預測")
    args = parser.parse_args()

    for dataset in args.dataset:
        if not args.summary_only:
            run(dataset, args)
        if csv_path(dataset).exists():
            per_fold(dataset)
        summarize(dataset, csv_path(dataset), BASELINE)


if __name__ == "__main__":
    main()
