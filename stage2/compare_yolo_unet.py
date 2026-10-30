import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

# 相對路徑以這個腳本所在資料夾為準
WEIGHTS = "runs/segment/20261008/yolov8n_seg_mf_559_unsharp_mask/weights/best.pt"
IMAGES_DIR = "runs/gt/images/um_test"
# GT 可以是 YOLO 多邊形標註(.txt)、LabelMe 標註(.json)或遮罩影像,都以檔名(不含副檔名)對應影像
GT_DIR = "runs/gt/labels"
# UNET 預測遮罩的資料夾,請填入實際路徑(或執行時用 --unet 指定)
UNET_DIR = "runs/unet/masks"
# UNET 遮罩檔名若比影像多一段後綴(例如 xxx_pred.png)就填 "_pred"
UNET_SUFFIX = ""

NAMES = {0: "mf", 1: "ian"}
# 遮罩影像中各類別的像素值(0 為背景);GT 用 .txt 標註時 GT_VALUES 不會用到
GT_VALUES = {"mf": 1, "ian": 2}
UNET_VALUES = {"mf": 1, "ian": 2}

CONF = 0.4
IOU = 0.3
# None 表示沿用權重訓練時的 imgsz
IMGSZ = None

OUT_DIR = ROOT / "compare_results" / "yolo_vs_unet"
PER_IMAGE_CSV = OUT_DIR / "per_image.csv"
SUMMARY_CSV = OUT_DIR / "summary.csv"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MODELS = ["yolo", "unet"]
METRICS = ["dice", "iou", "precision", "recall"]
FIELDS = ["filename", "model", "class"] + METRICS + ["tp", "fp", "fn", "gt_pixels", "pred_pixels"]
# 疊圖輪廓顏色(BGR)
COLORS = {"gt": (0, 255, 0), "yolo": (0, 0, 255), "unet": (255, 128, 0)}

resized = Counter()


def index_files(folder, suffix=""):
    # 回傳 {檔名(去掉後綴): 路徑},只收影像檔
    files = {}
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in IMAGE_SUFFIXES or not p.stem.endswith(suffix):
            continue
        stem = p.stem[: len(p.stem) - len(suffix)]
        if stem in files:
            raise SystemExit(f"{folder}: 檔名重複 {stem}\n  {files[stem]}\n  {p}")
        files[stem] = p
    return files


def read_mask(path, values, shape, source):
    # 讀取遮罩影像並拆成 {類別: bool 陣列},尺寸與原圖不同時以最近鄰縮放到原圖大小
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise SystemExit(f"無法讀取遮罩: {path}")
    if mask.ndim == 3:
        mask = mask[..., :3]
        if not (mask == mask[..., :1]).all():
            raise SystemExit(f"{path}: 是彩色遮罩,請先轉成單通道的類別索引圖")
        mask = mask[..., 0]
    unknown = sorted(set(np.unique(mask).tolist()) - set(values.values()) - {0})
    if unknown:
        raise SystemExit(
            f"{path}: 出現未定義的像素值 {unknown[:10]}\n"
            f"  目前 {source.upper()}_VALUES = {values},請依遮罩實際的像素值修改"
        )
    if mask.shape != shape:
        resized[source] += 1
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return {cls: mask == value for cls, value in values.items()}


def load_gt(stem, shape, gt_dir, gt_files):
    txt = gt_dir / f"{stem}.txt"
    labelme = gt_dir / f"{stem}.json"
    h, w = shape
    masks = {cls: np.zeros(shape, dtype=np.uint8) for cls in NAMES.values()}

    if not txt.exists() and labelme.exists():
        # LabelMe 標註: 座標為像素,依標註時的影像尺寸換算到目前影像
        data = json.loads(labelme.read_text())
        scale = (w / data["imageWidth"], h / data["imageHeight"])
        for s in data["shapes"]:
            if s["shape_type"] != "polygon":
                raise SystemExit(f"{labelme}: 不支援的標註形狀 {s['shape_type']}(只支援 polygon)")
            if s["label"] not in masks:
                raise SystemExit(f"{labelme}: 類別 {s['label']} 不在 NAMES {NAMES} 中")
            points = np.array(s["points"], dtype=np.float32) * scale
            cv2.fillPoly(masks[s["label"]], [np.round(points).astype(np.int32)], 1)
        return {cls: m.astype(bool) for cls, m in masks.items()}

    if not txt.exists():
        if stem not in gt_files:
            raise SystemExit(f"{gt_dir}: 找不到 {stem} 的 GT(.txt、LabelMe .json 或遮罩影像)")
        return read_mask(gt_files[stem], GT_VALUES, shape, "gt")

    # YOLO 多邊形標註: 每行為 class x1 y1 x2 y2 ...(座標已正規化)
    for line in txt.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 7 or len(parts) % 2 == 0:
            raise SystemExit(f"{txt}: 不是多邊形標註(可能是 bbox 格式)\n  {line}")
        if int(parts[0]) not in NAMES:
            raise SystemExit(f"{txt}: 類別 {parts[0]} 不在 NAMES {NAMES} 中")
        points = np.array(parts[1:], dtype=np.float32).reshape(-1, 2) * (w, h)
        cv2.fillPoly(masks[NAMES[int(parts[0])]], [np.round(points).astype(np.int32)], 1)
    return {cls: m.astype(bool) for cls, m in masks.items()}


def yolo_masks(result, shape):
    # 把同一類別的所有 instance 合併成一張遮罩;沒有偵測到時為全黑
    masks = {cls: np.zeros(shape, dtype=bool) for cls in NAMES.values()}
    if result.masks is None:
        return masks
    data = result.masks.data.cpu().numpy() > 0.5
    for m, c in zip(data, result.boxes.cls.int().tolist()):
        if m.shape != shape:
            m = cv2.resize(m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        masks[NAMES[c]] |= m
    return masks


def pixel_metrics(pred, gt):
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    if tp + fp + fn == 0:
        # 預測與 GT 都沒有這個類別,視為完全正確
        dice = iou = precision = recall = 1.0
    else:
        dice = 2 * tp / (2 * tp + fp + fn)
        iou = tp / (tp + fp + fn)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
    row = {"dice": dice, "iou": iou, "precision": precision, "recall": recall}
    row = {k: round(v, 5) for k, v in row.items()}
    row.update(tp=tp, fp=fp, fn=fn, gt_pixels=tp + fn, pred_pixels=tp + fp)
    return row


def save_overlay(image, masks, path):
    # 在原圖上畫出 GT / YOLO / UNET 的輪廓(各類別合併)
    canvas = image.copy()
    thickness = max(1, round(max(canvas.shape[:2]) / 600))
    for i, (source, color) in enumerate(COLORS.items()):
        merged = np.zeros(canvas.shape[:2], dtype=np.uint8)
        for m in masks[source].values():
            merged |= m.astype(np.uint8)
        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, thickness)
        cv2.putText(canvas, source.upper(), (10, 30 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imwrite(str(path), canvas)


def run(args):
    images_dir = (ROOT / args.images).resolve()
    gt_dir = (ROOT / args.gt).resolve()
    unet_dir = (ROOT / args.unet).resolve()
    for name, folder in (("影像", images_dir), ("GT", gt_dir), ("UNET", unet_dir)):
        if not folder.is_dir():
            raise SystemExit(f"找不到{name}資料夾: {folder}")

    images = index_files(images_dir)
    if not images:
        raise SystemExit(f"{images_dir}: 找不到任何影像")
    gt_files = index_files(gt_dir)
    unet_files = index_files(unet_dir, UNET_SUFFIX)
    missing = sorted(set(images) - set(unet_files))
    if missing:
        raise SystemExit(
            f"UNET 遮罩與測試影像的檔名不一致: 共 {len(images)} 張影像,缺少 {len(missing)} 張的 UNET 遮罩\n"
            f"  缺少(前5): {missing[:5]}\n  UNET 檔名(前5): {sorted(unet_files)[:5]}"
        )

    model = YOLO(ROOT / args.weights)
    if dict(model.names) != NAMES:
        raise SystemExit(f"權重的類別 {dict(model.names)} 與 NAMES {NAMES} 不一致")

    mask_dirs = {cls: OUT_DIR / "yolo_masks" / cls for cls in NAMES.values()}
    for d in mask_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    overlay_dir = OUT_DIR / "overlays"
    if args.overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    # retina_masks 讓輸出的遮罩直接是原圖解析度
    predict_args = dict(source=str(images_dir), conf=args.conf, iou=args.iou, retina_masks=True, stream=True, verbose=False)
    if IMGSZ is not None:
        predict_args["imgsz"] = IMGSZ

    rows = []
    for result in model.predict(**predict_args):
        stem = Path(result.path).stem
        shape = tuple(result.orig_shape)
        masks = {
            "gt": load_gt(stem, shape, gt_dir, gt_files),
            "yolo": yolo_masks(result, shape),
            "unet": read_mask(unet_files[stem], UNET_VALUES, shape, "unet"),
        }
        for cls in NAMES.values():
            cv2.imwrite(str(mask_dirs[cls] / f"{stem}.png"), masks["yolo"][cls].astype(np.uint8) * 255)
            for m in MODELS:
                rows.append({"filename": stem, "model": m, "class": cls, **pixel_metrics(masks[m][cls], masks["gt"][cls])})
        if args.overlays:
            save_overlay(result.orig_img, masks, overlay_dir / f"{stem}.jpg")

    with open(PER_IMAGE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"共比對 {len(rows) // (len(MODELS) * len(NAMES))} 張影像")
    for source, n in resized.items():
        print(f"[注意] {source} 有 {n} 張遮罩尺寸與原圖不同,已用最近鄰縮放到原圖大小")


def wilcoxon_p(diffs):
    # 成對 Wilcoxon signed-rank 檢定;沒有安裝 scipy 或差異全為 0 時回傳空字串
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return ""
    if not any(diffs):
        return ""
    return f"{float(wilcoxon(diffs).pvalue):.2e}"


def summarize(per_image_csv=PER_IMAGE_CSV, summary_csv=SUMMARY_CSV):
    if not per_image_csv.exists():
        print("尚無結果")
        return
    with open(per_image_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    by_key = {}
    for r in rows:
        by_key.setdefault((r["class"], r["model"]), {})[r["filename"]] = r

    summary = []
    for cls in NAMES.values():
        for m in MODELS:
            runs = by_key.get((cls, m), {})
            if not runs:
                continue
            item = {"class": cls, "model": m, "n": len(runs)}
            for metric in METRICS:
                values = [float(r[metric]) for r in runs.values()]
                item[f"{metric}_mean"] = round(statistics.mean(values), 4)
                item[f"{metric}_std"] = round(statistics.stdev(values), 4) if len(values) > 1 else 0.0
            # 把所有影像的像素加總後再算一次(不受小目標影像的極端值影響)
            tp, fp, fn = (sum(int(r[k]) for r in runs.values()) for k in ("tp", "fp", "fn"))
            item["dice_global"] = round(2 * tp / (2 * tp + fp + fn), 4) if tp + fp + fn else 1.0
            item["iou_global"] = round(tp / (tp + fp + fn), 4) if tp + fp + fn else 1.0
            # GT 有這個類別但完全沒預測到的影像數
            item["missed"] = sum(int(r["gt_pixels"]) > 0 and int(r["pred_pixels"]) == 0 for r in runs.values())
            summary.append(item)

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print("\n逐張影像的平均 ± 標準差;global 為全部像素加總後計算;missed 為 GT 有但完全沒預測到的影像數")
    print(f"{'class':<8}{'model':<8}{'n':>4}" + "".join(f"{m:>20}" for m in METRICS) + f"{'dice_global':>13}{'iou_global':>12}{'missed':>8}")
    for item in summary:
        cells = "".join(f"{item[f'{m}_mean']:>12.4f} ±{item[f'{m}_std']:<6.4f}" for m in METRICS)
        print(f"{item['class']:<8}{item['model']:<8}{item['n']:>4}{cells}{item['dice_global']:>13.4f}{item['iou_global']:>12.4f}{item['missed']:>8}")

    print("\nYOLO 與 UNET 的成對差異(同一張影像相減,正值代表 YOLO 較高;wins = YOLO 贏過 UNET 的張數)")
    print(f"{'class':<8}{'metric':<8}{'diff':>20}{'wins':>10}{'ties':>7}{'wilcoxon p':>13}")
    for cls in NAMES.values():
        yolo, unet = by_key.get((cls, "yolo"), {}), by_key.get((cls, "unet"), {})
        keys = sorted(set(yolo) & set(unet))
        for metric in ("dice", "iou"):
            diffs = [float(yolo[k][metric]) - float(unet[k][metric]) for k in keys]
            if not diffs:
                continue
            std = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
            text = f"{statistics.mean(diffs):+.4f} ±{std:.4f}"
            wins = f"{sum(d > 0 for d in diffs)}/{len(diffs)}"
            print(f"{cls:<8}{metric:<8}{text:>20}{wins:>10}{sum(d == 0 for d in diffs):>7}{str(wilcoxon_p(diffs)) or '-':>13}")
    print(f"\n逐張結果: {per_image_csv}\n彙總結果: {summary_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=WEIGHTS)
    parser.add_argument("--images", default=IMAGES_DIR)
    parser.add_argument("--gt", default=GT_DIR)
    parser.add_argument("--unet", default=UNET_DIR, help="UNET 預測遮罩的資料夾")
    parser.add_argument("--conf", type=float, default=CONF)
    parser.add_argument("--iou", type=float, default=IOU)
    parser.add_argument("--overlays", action="store_true", help="另外輸出 GT / YOLO / UNET 輪廓疊圖")
    parser.add_argument("--summary-only", action="store_true", help="只重新彙總已有的 per_image.csv,不重新預測")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.summary_only:
        if args.unet is None:
            parser.error("請在 UNET_DIR 填入 UNET 遮罩資料夾,或用 --unet 指定")
        run(args)

    summarize()
    print(f"YOLO 遮罩: {OUT_DIR / 'yolo_masks'}")


if __name__ == "__main__":
    main()
