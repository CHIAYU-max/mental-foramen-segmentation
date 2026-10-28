import argparse
import csv
import statistics
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from compare_yolo_unet import IOU, NAMES, ROOT, load_gt
from eval_kfold_variants import MIN_CONF, csv_path, yolo_variants
from kfold_yolo_unet import K_FOLDS, OUT_DIR, PROJECT

# 以 eval_kfold_variants.py 的交叉驗證結果找出預測最差的影像,重新預測後畫成疊圖
VARIANT = "yolo_top1"
TOP_N = 30
# 影像很小,放大後比較好看
SCALE = 2
# 輪廓顏色(BGR)
COLORS = {
    ("gt", "mf"): (0, 255, 0),
    ("gt", "ian"): (255, 255, 0),
    ("pred", "mf"): (0, 0, 255),
    ("pred", "ian"): (255, 0, 255),
}
LEGEND = [("GT mf", ("gt", "mf")), ("GT ian", ("gt", "ian")), ("pred mf", ("pred", "mf")), ("pred ian", ("pred", "ian"))]


def category(row):
    dice, tp, gt, pred = float(row["dice"]), int(row["tp"]), int(row["gt_pixels"]), int(row["pred_pixels"])
    if gt > 0 and pred == 0:
        return "漏檢(沒有預測)"
    if pred > 0 and tp == 0:
        return "位置錯誤(完全沒重疊)"
    if dice < 0.5:
        return "重疊不足(Dice<0.5)"
    if dice < 0.8:
        return "尚可(0.5~0.8)"
    return "良好(>=0.8)"


def breakdown(by_class):
    # 只用已有的逐張結果統計失敗類型,不需要重新預測
    order = ["漏檢(沒有預測)", "位置錯誤(完全沒重疊)", "重疊不足(Dice<0.5)", "尚可(0.5~0.8)", "良好(>=0.8)"]
    for cls, rows in by_class.items():
        print(f"\n[{cls}] {len(rows)} 張的結果分布")
        groups = {name: [] for name in order}
        for r in rows.values():
            groups[category(r)].append(r)
        for name in order:
            items = groups[name]
            text = f"  {name:<22}{len(items):>5} 張 ({len(items) / len(rows):6.1%})"
            # 預測面積 / GT 面積: 大於 1 是畫太大,小於 1 是畫太小
            ratios = [int(r["pred_pixels"]) / int(r["gt_pixels"]) for r in items if int(r["gt_pixels"]) > 0 and int(r["pred_pixels"]) > 0]
            if ratios:
                text += f"   預測/GT 面積比中位數 {statistics.median(ratios):.2f}"
            print(text)


def draw(image, gt, pred, header):
    h, w = image.shape[:2]
    size = (w * SCALE, h * SCALE)
    clean = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
    overlay = clean.copy()
    for (source, cls), color in COLORS.items():
        mask = (gt if source == "gt" else pred)[cls].astype(np.uint8)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 1)

    # 左: 原圖, 右: 疊圖, 上方為檔名與分數, 下方為圖例
    body = np.hstack([clean, overlay])
    top = np.zeros((28, body.shape[1], 3), dtype=np.uint8)
    cv2.putText(top, header, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    bottom = np.zeros((28, body.shape[1], 3), dtype=np.uint8)
    for i, (text, key) in enumerate(LEGEND):
        cv2.putText(bottom, text, (6 + i * 130, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[key], 1, cv2.LINE_AA)
    return np.vstack([top, body, bottom])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default=VARIANT, help="只支援 YOLO 的版本(yolo_top1 或 yolo_conf0.4 等)")
    parser.add_argument("--top", type=int, default=TOP_N, help="每個類別輸出 Dice 最低的前幾張")
    parser.add_argument("--stats-only", action="store_true", help="只印出失敗類型統計,不產生疊圖")
    args = parser.parse_args()

    if not csv_path("cv").exists():
        raise SystemExit(f"找不到 {csv_path('cv')},請先執行 eval_kfold_variants.py --dataset cv")
    with open(csv_path("cv"), newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["variant"] == args.variant]
    if not rows:
        raise SystemExit(f"{csv_path('cv')} 中沒有 {args.variant} 的結果")

    by_class = {cls: {} for cls in NAMES.values()}
    for r in rows:
        by_class[r["class"]][r["filename"]] = r
    print(f"版本: {args.variant}")
    breakdown(by_class)
    if args.stats_only:
        return

    # 每個類別取 Dice 最低的前 N 張;{檔名: [(類別, 名次)]}
    selected = {}
    for cls, items in by_class.items():
        worst = sorted(items.values(), key=lambda r: float(r["dice"]))[: args.top]
        for rank, r in enumerate(worst, start=1):
            selected.setdefault(r["filename"], []).append((cls, rank))

    out_dir = OUT_DIR / "failures" / args.variant
    for cls in NAMES.values():
        (out_dir / cls).mkdir(parents=True, exist_ok=True)

    for fold in range(K_FOLDS):
        val_txt = OUT_DIR / "yaml" / f"fold{fold}_val.txt"
        paths = [Path(line) for line in val_txt.read_text().splitlines() if line and Path(line).stem in selected]
        if not paths:
            continue
        source = OUT_DIR / "yaml" / f"failures_fold{fold}.txt"
        source.write_text("".join(f"{p}\n" for p in paths))
        gt_dirs = {p.stem: p.parent.parent / "labels" for p in paths}

        yolo = YOLO(ROOT / "runs" / "segment" / PROJECT / f"yolo_fold{fold}" / "weights" / "best.pt")
        results = yolo.predict(source=str(source), conf=MIN_CONF, iou=IOU, max_det=50, retina_masks=True, stream=True, verbose=False)
        for result in results:
            stem = Path(result.path).stem
            shape = tuple(result.orig_shape)
            gt = load_gt(stem, shape, gt_dirs[stem], {})
            pred = yolo_variants(result, shape)[args.variant]

            # 各類別信心最高的預測的信心值,沒有預測時為 0
            confs = {cls: 0.0 for cls in NAMES.values()}
            for c, conf in zip(result.boxes.cls.int().tolist(), result.boxes.conf.tolist()):
                confs[NAMES[c]] = max(confs[NAMES[c]], conf)
            scores = " | ".join(f"{cls} dice {float(by_class[cls][stem]['dice']):.2f} conf {confs[cls]:.2f}" for cls in NAMES.values())
            image = draw(result.orig_img, gt, pred, f"{stem} fold{fold} | {scores}")

            for cls, rank in selected[stem]:
                dice = float(by_class[cls][stem]["dice"])
                cv2.imwrite(str(out_dir / cls / f"{rank:03d}_dice{dice:.2f}_{stem}.jpg"), image)
        print(f"fold {fold}: 已輸出 {len(paths)} 張")

    print(f"\n疊圖已輸出到: {out_dir}\n  檔名開頭的編號是名次(001 最差),每個類別各 {args.top} 張")


if __name__ == "__main__":
    main()
