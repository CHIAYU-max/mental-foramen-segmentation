"""驗證 ian 是否有助於 mf:比較「mf+ian」與「只訓練 mf」,並測試 mf 的挑選規則。

沿用 compare_preprocessing.py 產生的 5 折切分與 original 的訓練結果。
"""

import argparse
import csv
import gc
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

# 沿用的 5 折結果(mf+ian 模型與切分)
BASE_DIR = ROOT / "compare_results" / "compare_mf_397_5fold"
BASE_VARIANT = "original"

MF, IAN = 0, 1
K_FOLDS = 5
SEED = 0
MODEL = "yolov8n-seg.pt"
PROJECT = "compare_mf_context"
TRAIN_ARGS = dict(epochs=200, patience=50, imgsz=1280, batch=16, device=0, workers=8)

# 挑選規則的參數(事先固定,沒有依結果調整)
CONF_DEFAULT = 0.25  # 一般輸出門檻
CONF_MIN = 0.05  # 候選 mf 的最低信心
PRED_CONF = 0.01  # 推論時保留的最低信心
PRED_BATCH = 4  # 評估時的批次大小(調小以節省 GPU 記憶體,不影響結果)
NEAR_RATIO = 0.05  # 「靠近 ian」= 距離 ian 遮罩不超過影像對角線的 5%
IOU_THRES = 0.5  # 判定命中的 IoU 門檻

OUT_DIR = ROOT / "compare_results" / PROJECT
TRAIN_CSV = OUT_DIR / "train_runs.csv"
EVAL_CSV = OUT_DIR / "eval_by_fold.csv"
SUMMARY_CSV = OUT_DIR / "summary.csv"

RULES = ["conf>=0.25", "top1", "ian_guided"]
COUNTS = ["n_gt", "tp_mask", "fp_mask", "fn_mask", "tp_box", "fp_box", "fn_box", "iou_sum", "dice_sum"]


def base_weights():
    # 從 compare_preprocessing.py 的 runs.csv 找出 original 各折的 best.pt
    with open(BASE_DIR / "runs.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["variant"] == BASE_VARIANT and int(r["seed"]) == SEED]
    return {int(r["fold"]): Path(r["save_dir"]) / "weights" / "best.pt" for r in rows}


def read_train_runs():
    if not TRAIN_CSV.exists():
        return {}
    with open(TRAIN_CSV, newline="") as f:
        return {int(r["fold"]): Path(r["save_dir"]) for r in csv.DictReader(f)}


def train_mf_only(fold):
    # classes=[MF] 會讓訓練與驗證都只使用 mf 的標註,其餘設定與 mf+ian 完全相同
    data_yaml = BASE_DIR / "yaml" / f"{BASE_VARIANT}_fold{fold}.yaml"
    model = YOLO(MODEL)
    model.train(
        data=str(data_yaml),
        project=PROJECT,
        name=f"mf_only_fold{fold}_seed{SEED}",
        seed=SEED,
        classes=[MF],
        save=True,
        plots=True,
        **TRAIN_ARGS,
    )
    save_dir = Path(model.trainer.save_dir)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    new_file = not TRAIN_CSV.exists()
    with open(TRAIN_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["fold", "save_dir"])
        writer.writerow([fold, save_dir])
    return save_dir


def load_gt_mf(img_path, h, w):
    # 讀取 YOLO 分割標註,回傳 mf 的 [(遮罩, 框)]
    parts = list(Path(img_path).parts)
    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    label_path = Path(*parts).with_suffix(".txt")
    gts = []
    if not label_path.exists():
        return gts
    for line in label_path.read_text().splitlines():
        vals = line.split()
        if not vals or int(float(vals[0])) != MF:
            continue
        coords = np.array(vals[1:], dtype=np.float32)
        if len(coords) == 4:  # 只有框的標註: cx cy w h
            cx, cy, bw, bh = coords
            pts = np.array([[cx - bw / 2, cy - bh / 2], [cx + bw / 2, cy - bh / 2], [cx + bw / 2, cy + bh / 2], [cx - bw / 2, cy + bh / 2]])
        else:
            pts = coords.reshape(-1, 2)
        pts = pts * np.array([w, h])
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
        box = np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
        gts.append((mask.astype(bool), box))
    return gts


def mask_iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def box_iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def parse_result(result):
    # 回傳依信心由高到低排序的 mf、ian 預測: [{"conf", "box", "mask"}]
    h, w = result.orig_shape
    preds = {MF: [], IAN: []}
    if result.masks is None or len(result.boxes) == 0:
        return preds
    masks = result.masks.data.cpu().numpy() > 0.5
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    for mask, box, conf, cls in zip(masks, boxes, confs, classes):
        if cls not in preds:
            continue
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        preds[cls].append({"conf": float(conf), "box": box, "mask": mask})
    for cls in preds:
        preds[cls].sort(key=lambda p: p["conf"], reverse=True)
    return preds


def select_mf(rule, mf, ian, h, w):
    if rule == "conf>=0.25":
        return [p for p in mf if p["conf"] >= CONF_DEFAULT]
    if rule == "top1":
        return mf[:1]
    # ian_guided: 在信心夠高的候選中,取「靠近 ian 遮罩」且信心最高者;沒有則退回 top1
    ian = [p for p in ian if p["conf"] >= CONF_DEFAULT and p["mask"].any()]
    if not ian or not mf:
        return mf[:1]
    dist = cv2.distanceTransform((~ian[0]["mask"]).astype(np.uint8), cv2.DIST_L2, 3)
    limit = NEAR_RATIO * float(np.hypot(h, w))
    for p in mf:
        if p["conf"] < CONF_MIN:
            break
        cx = int(np.clip((p["box"][0] + p["box"][2]) / 2, 0, w - 1))
        cy = int(np.clip((p["box"][1] + p["box"][3]) / 2, 0, h - 1))
        if dist[cy, cx] <= limit:
            return [p]
    return mf[:1]


def score_image(selected, gts):
    # 每個標註最多配對一個預測(依信心由高到低),IoU 達門檻才算命中
    out = dict.fromkeys(COUNTS, 0.0)
    out["n_gt"] = len(gts)
    for kind, iou_fn, key in (("mask", mask_iou, 0), ("box", box_iou, 1)):
        matched = set()
        tp = 0
        for p in selected:
            ious = [(iou_fn(p[kind], gt[key]), i) for i, gt in enumerate(gts) if i not in matched]
            if ious and max(ious)[0] >= IOU_THRES:
                matched.add(max(ious)[1])
                tp += 1
        out[f"tp_{kind}"] = tp
        out[f"fp_{kind}"] = len(selected) - tp
        out[f"fn_{kind}"] = len(gts) - tp
    # 遮罩品質: 每個標註取與入選預測的最佳 IoU / Dice,沒有預測記 0
    for gt_mask, _ in gts:
        best = max((mask_iou(p["mask"], gt_mask) for p in selected), default=0.0)
        out["iou_sum"] += best
        out["dice_sum"] += 2 * best / (1 + best)
    return out


def evaluate_model(model_name, weights, fold):
    val_txt = BASE_DIR / "yaml" / f"{BASE_VARIANT}_fold{fold}_val.txt"
    images = [line for line in val_txt.read_text().splitlines() if line.strip()]
    rules = RULES if model_name == "mf+ian" else RULES[:2]
    totals = {rule: dict.fromkeys(COUNTS, 0.0) for rule in rules}

    model = YOLO(weights)
    results = model.predict(
        source=images,
        imgsz=TRAIN_ARGS["imgsz"],
        batch=PRED_BATCH,
        device=TRAIN_ARGS["device"],
        conf=PRED_CONF,
        retina_masks=True,
        stream=True,
        verbose=False,
    )
    for img_path, result in zip(images, results):
        h, w = result.orig_shape
        gts = load_gt_mf(img_path, h, w)
        preds = parse_result(result)
        for rule in rules:
            scores = score_image(select_mf(rule, preds[MF], preds[IAN], h, w), gts)
            for k in COUNTS:
                totals[rule][k] += scores[k]

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return [{"model": model_name, "rule": rule, "fold": fold, **totals[rule]} for rule in rules]


def summarize(rows):
    pooled = {}
    for r in rows:
        item = pooled.setdefault((r["model"], r["rule"]), dict.fromkeys(COUNTS, 0.0))
        for k in COUNTS:
            item[k] += float(r[k])

    summary = []
    for (model_name, rule), c in pooled.items():
        n = c["n_gt"] or 1
        summary.append(
            {
                "model": model_name,
                "rule": rule,
                "n_gt": int(c["n_gt"]),
                "tp_mask": int(c["tp_mask"]),
                "fp_mask": int(c["fp_mask"]),
                "fn_mask": int(c["fn_mask"]),
                "recall_mask": round(c["tp_mask"] / n, 4),
                "precision_mask": round(c["tp_mask"] / max(c["tp_mask"] + c["fp_mask"], 1), 4),
                "tp_box": int(c["tp_box"]),
                "fp_box": int(c["fp_box"]),
                "fn_box": int(c["fn_box"]),
                "recall_box": round(c["tp_box"] / n, 4),
                "precision_box": round(c["tp_box"] / max(c["tp_box"] + c["fp_box"], 1), 4),
                "mean_mask_iou": round(c["iou_sum"] / n, 4),
                "mean_dice": round(c["dice_sum"] / n, 4),
            }
        )

    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print(f"\nmf 結果({K_FOLDS} 折合計;命中 = IoU >= {IOU_THRES})")
    header = f"{'model':<10}{'rule':<13}{'n':>5} |{'TP':>5}{'FP':>5}{'FN':>5}{'recall':>8}{'prec':>8} |{'TP':>5}{'FP':>5}{'FN':>5}{'recall':>8}{'prec':>8} |{'mIoU':>7}{'Dice':>7}"
    print(f"{'':<28} |{'遮罩':^29} |{'框':^30} |")
    print(header)
    for s in summary:
        print(
            f"{s['model']:<10}{s['rule']:<13}{s['n_gt']:>5} |"
            f"{s['tp_mask']:>5}{s['fp_mask']:>5}{s['fn_mask']:>5}{s['recall_mask']:>8.3f}{s['precision_mask']:>8.3f} |"
            f"{s['tp_box']:>5}{s['fp_box']:>5}{s['fn_box']:>5}{s['recall_box']:>8.3f}{s['precision_box']:>8.3f} |"
            f"{s['mean_mask_iou']:>7.3f}{s['mean_dice']:>7.3f}"
        )
    print(f"\n各折結果: {EVAL_CSV}\n彙總結果: {SUMMARY_CSV}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(K_FOLDS)))
    parser.add_argument("--eval-only", action="store_true", help="不訓練,只評估已有的模型")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    weights_both = base_weights()
    missing = [f for f in args.folds if f not in weights_both or not weights_both[f].exists()]
    if missing:
        parser.error(f"找不到 {BASE_VARIANT} 第 {missing} 折的 best.pt,請先完成 compare_preprocessing.py")

    # 1. 只訓練 mf(已完成的折不重跑,中斷後可直接重新執行接續)
    mf_only = read_train_runs()
    if not args.eval_only:
        for fold in args.folds:
            if fold in mf_only and (mf_only[fold] / "weights" / "best.pt").exists():
                print(f"[已完成] mf_only fold={fold}")
                continue
            mf_only[fold] = train_mf_only(fold)

    # 2. 兩種模型在同一批驗證影像上評估 mf
    rows = []
    for fold in args.folds:
        rows += evaluate_model("mf+ian", weights_both[fold], fold)
        if fold in mf_only and (mf_only[fold] / "weights" / "best.pt").exists():
            rows += evaluate_model("mf_only", mf_only[fold] / "weights" / "best.pt", fold)
        else:
            print(f"[略過] mf_only fold={fold}: 尚未訓練")

    with open(EVAL_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "rule", "fold"] + COUNTS)
        writer.writeheader()
        writer.writerows(rows)
    summarize(rows)


if __name__ == "__main__":
    main()
