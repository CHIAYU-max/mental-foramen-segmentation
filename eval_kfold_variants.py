import argparse
import csv
import statistics
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from compare_yolo_unet import GT_DIR, IMAGES_DIR, IOU, METRICS, NAMES, ROOT, index_files, load_gt, pixel_metrics, wilcoxon_p
from kfold_yolo_unet import K_FOLDS, OUT_DIR, PROJECT, UNet, predict_masks
from predict_unet import DEVICE

# 用 kfold_yolo_unet.py 各折已訓練好的權重重新預測,不重新訓練
#   cv:   每一折的模型預測自己那一折的驗證影像
#   test: 每一折的模型都預測一次獨立測試集,逐張取 5 個模型的平均
# YOLO 以很低的門檻預測一次,再由同一批結果產生各種門檻的版本
MIN_CONF = 0.01
CONFS = [0.05, 0.1, 0.15, 0.25, 0.4]
BASELINE = "yolo_conf0.4"
# 互補版本: YOLO 該類別沒有預測時改用 U-Net 的遮罩
HYBRID_BASES = [BASELINE, "yolo_top1"]

FIELDS = ["dataset", "fold", "filename", "variant", "class"] + METRICS + ["tp", "fp", "fn", "gt_pixels", "pred_pixels"]


def csv_path(dataset):
    return OUT_DIR / f"variants_{dataset}.csv"


def yolo_variants(result, shape):
    # 回傳 {版本: {類別: bool 遮罩}}
    instances = []
    if result.masks is not None:
        data = result.masks.data.cpu().numpy() > 0.5
        for m, c, conf in zip(data, result.boxes.cls.int().tolist(), result.boxes.conf.tolist()):
            if m.shape != shape:
                m = cv2.resize(m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            instances.append((NAMES[c], conf, m))

    variants = {}
    for threshold in CONFS:
        masks = {cls: np.zeros(shape, dtype=bool) for cls in NAMES.values()}
        for cls, conf, m in instances:
            if conf >= threshold:
                masks[cls] |= m
        variants[f"yolo_conf{threshold}"] = masks

    # 每張影像每個類別只有一個目標,直接取信心最高的那一個,不設門檻
    masks = {cls: np.zeros(shape, dtype=bool) for cls in NAMES.values()}
    for cls in NAMES.values():
        candidates = [(conf, m) for c, conf, m in instances if c == cls]
        if candidates:
            masks[cls] = max(candidates, key=lambda x: x[0])[1]
    variants["yolo_top1"] = masks
    return variants


def evaluate(dataset, fold, samples):
    # samples: [(影像路徑, 標註資料夾)]
    yolo_weights = ROOT / "runs" / "segment" / PROJECT / f"yolo_fold{fold}" / "weights" / "best.pt"
    unet_weights = OUT_DIR / "unet_weights" / f"unet_fold{fold}.pth"
    for path in (yolo_weights, unet_weights):
        if not path.exists():
            raise SystemExit(f"找不到 fold {fold} 的權重: {path}")

    unet = UNet(len(NAMES) + 1).to(DEVICE)
    unet.load_state_dict(torch.load(unet_weights, map_location=DEVICE)["model_state_dict"])
    unet.eval()

    source = OUT_DIR / "yaml" / f"variants_{dataset}_fold{fold}.txt"
    source.write_text("".join(f"{path}\n" for path, _ in samples))
    gt_dirs = {path.stem: gt_dir for path, gt_dir in samples}

    yolo = YOLO(yolo_weights)
    results = yolo.predict(source=str(source), conf=MIN_CONF, iou=IOU, max_det=50, retina_masks=True, stream=True, verbose=False)
    rows = []
    for result in results:
        stem = Path(result.path).stem
        shape = tuple(result.orig_shape)
        gt = load_gt(stem, shape, gt_dirs[stem], {})

        variants = yolo_variants(result, shape)
        variants["unet"] = predict_masks(unet, cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2GRAY))
        for base in HYBRID_BASES:
            variants[f"{base}+unet"] = {
                cls: variants[base][cls] if variants[base][cls].any() else variants["unet"][cls] for cls in NAMES.values()
            }

        for name, masks in variants.items():
            for cls in NAMES.values():
                row = {"dataset": dataset, "fold": fold, "filename": stem, "variant": name, "class": cls}
                rows.append({**row, **pixel_metrics(masks[cls], gt[cls])})
    print(f"[{dataset}] fold {fold}: {len(samples)} 張")
    return rows


def run(dataset, args):
    rows = []
    if dataset == "cv":
        for fold in range(K_FOLDS):
            val_txt = OUT_DIR / "yaml" / f"fold{fold}_val.txt"
            if not val_txt.exists():
                raise SystemExit(f"找不到 {val_txt},請先跑完 kfold_yolo_unet.py")
            paths = [Path(line) for line in val_txt.read_text().splitlines() if line]
            rows += evaluate(dataset, fold, [(p, p.parent.parent / "labels") for p in paths])
    else:
        images = index_files((ROOT / args.images).resolve())
        if not images:
            raise SystemExit(f"{args.images}: 找不到任何影像")
        gt_dir = (ROOT / args.gt).resolve()
        for fold in range(K_FOLDS):
            rows += evaluate(dataset, fold, [(p, gt_dir) for p in images.values()])

    with open(csv_path(dataset), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summarize(dataset, per_image_csv=None, baseline=BASELINE):
    per_image_csv = per_image_csv or csv_path(dataset)
    if not per_image_csv.exists():
        print(f"[{dataset}] 尚無結果")
        return
    with open(per_image_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    # {(類別, 版本): {檔名: [各折模型的結果]}}
    groups = {}
    for r in rows:
        groups.setdefault((r["class"], r["variant"]), {}).setdefault(r["filename"], []).append(r)
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    n_models = len({r["fold"] for r in rows}) if dataset == "test" else 1

    def per_image(cls, variant, metric):
        # 逐張取各折模型的平均(cv 每張只有一個模型)
        return {name: statistics.mean(float(r[metric]) for r in rs) for name, rs in groups[(cls, variant)].items()}

    title = "交叉驗證(每張影像由沒看過它的那一折模型預測)" if dataset == "cv" else f"獨立測試集(逐張取 {n_models} 個模型的平均)"
    print(f"\n===== {title} =====")
    print("missed 為 GT 有但完全沒預測到的影像數(test 為各模型的平均);diff 與 p 為和 " + baseline + " 的逐張成對比較")
    print(f"{'class':<6}{'variant':<22}{'n':>5}" + "".join(f"{m:>20}" for m in METRICS) + f"{'missed':>9}{'dice diff':>11}{'wilcoxon p':>12}")
    summary = []
    for cls in NAMES.values():
        base = per_image(cls, baseline, "dice")
        table = []
        for variant in variants:
            # 只預測部分類別的版本,在其他類別的表中不列出
            if (cls, variant) not in groups:
                continue
            item = {"dataset": dataset, "class": cls, "variant": variant, "n": len(groups[(cls, variant)])}
            for metric in METRICS:
                values = list(per_image(cls, variant, metric).values())
                item[f"{metric}_mean"] = round(statistics.mean(values), 4)
                item[f"{metric}_std"] = round(statistics.stdev(values), 4) if len(values) > 1 else 0.0
            records = [r for rs in groups[(cls, variant)].values() for r in rs]
            item["missed"] = round(sum(int(r["gt_pixels"]) > 0 and int(r["pred_pixels"]) == 0 for r in records) / n_models, 1)
            dice = per_image(cls, variant, "dice")
            diffs = [dice[k] - base[k] for k in dice]
            item["dice_diff"] = round(statistics.mean(diffs), 4)
            item["wilcoxon_p"] = "" if variant == baseline else wilcoxon_p(diffs)
            table.append(item)
        table.sort(key=lambda x: x["dice_mean"], reverse=True)
        for item in table:
            cells = "".join(f"{item[f'{m}_mean']:>12.4f} ±{item[f'{m}_std']:<6.4f}" for m in METRICS)
            print(f"{cls:<6}{item['variant']:<22}{item['n']:>5}{cells}{item['missed']:>9}{item['dice_diff']:>+11.4f}{item['wilcoxon_p'] or '-':>12}")
        summary += table

    summary_csv = per_image_csv.with_name(f"{per_image_csv.stem}_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"逐張結果: {per_image_csv}\n彙總結果: {summary_csv}")


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
        summarize(dataset)


if __name__ == "__main__":
    main()
