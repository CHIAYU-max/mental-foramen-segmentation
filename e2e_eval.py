import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.stats import binomtest, wilcoxon
from ultralytics import YOLO

from compare_yolo_unet import IOU, NAMES, ROOT, load_gt
from eval_kfold_variants import MIN_CONF
from kfold_pano import EXCLUDE_TEST, PROJECT, ROI_SOURCES, yolo_dir
from kfold_patient import UNET_TAG
from kfold_unet_variants import CONFIGS, build_model, predict_masks
from mf_position import ITEMS_JSON, KEY_CSV, centroid, classify
from mf_position_eval import lines_from_item
from mf_unsharp_mask import unsharp_mask
from paper_metrics import image_metrics, read_masks
from pipeline_timing import PANO_DIR, TEETH_MODEL, UNSHARP
from predict_unet import DEVICE
from rerank_candidates import get_candidates
from stage1_eval import MIN_GAP, MIN_SCORE, Sized, auto_rois

# 第一階段 E: 端到端評估(2026-10-25)。全景片 → 第一階段自動裁 ROI(原圖、imgsz 1024,stage1_eval 的決定)
#   → unsharp mask → 第二階段五折多數決,與「人工 ROI + 同一組五折模型」比較,量化第一階段造成的損失。
# 對象: kfold_pano 測試集的 ROI(排除 EXCLUDE_TEST),且 roi_sources 比對可信(與 stage1_eval 相同條件)。
# GT: 人工 ROI 的 mf / ian 多邊形 → 全景片座標 → 自動 ROI 座標(超出自動 ROI 的部分被裁掉)。
#   人工 ROI 的檔名帶 Left / Right,自動 ROI 取同一側;該側沒有自動 ROI → mf 算沒命中、Dice 0。
# 第二階段: 推論規則與 kfold_pano.predict_fold 相同(U-Net predict_masks;YOLO conf 0.01、各類別取信心最高的一個),
#   五個 fold 的模型逐像素多數決。人工 ROI 這一側直接讀 kfold_pano 存好的遮罩。
# 指標: paper_metrics.image_metrics(mf_dice、mf_hit_1R、ian_dice...),成對比較: Dice 用 Wilcoxon、命中用 McNemar 精確檢定。
# 牙位: 預測頦孔換回全景片座標後用 mf_position.classify(參考線取自 items.json),與人工 ROI 的預測、GT 頦孔的牙位比較。
OUT = ROOT / "compare_results" / "e2e"
MASKS = ROOT / "compare_results" / PROJECT
FOLDS = range(5)
GT_LABELS = ROOT / "runs" / "gt" / "labels"
GT_IMAGES = ROOT / "runs" / "gt" / "images" / "um_test"
KEYS = ["mf_dice", "mf_hit_1R", "ian_dice", "mf_predicted"]


def jpeg(img, quality=95):
    # 模擬存成 JPEG 再讀回(cv2.imwrite 預設品質 95)。當初的 ROI 經過兩次: fdi_cropped 存裁切、mf_unsharp_mask 存銳化結果
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1], cv2.IMREAD_COLOR)


def test_rois():
    # {來源全景片: [(stem, side, x, y, w, h)]}
    rois = defaultdict(list)
    with open(ROI_SOURCES, newline="") as f:
        for r in csv.DictReader(f):
            if r["set"] != "test" or r["stem"] in EXCLUDE_TEST or not r["source_pano"] or float(r["score"]) < MIN_SCORE:
                continue
            if r["second_score"] and float(r["score"]) - float(r["second_score"]) < MIN_GAP:
                continue
            side = "Left" if "_Left_" in r["stem"] else "Right"
            rois[r["source_pano"]].append((r["stem"], side, *(int(r[k]) for k in "xywh")))
    return rois


def gt_polygons(stem, x, y, w, h):
    # {類別: [全景片座標的多邊形]}
    polys = defaultdict(list)
    for line in (GT_LABELS / f"{stem}.txt").read_text().splitlines():
        parts = line.split()
        if parts:
            polys[NAMES[int(parts[0])]].append(np.array(parts[1:], dtype=float).reshape(-1, 2) * (w, h) + (x, y))
    return polys


class Stage2:
    # 五個 fold 的第二階段模型,回傳多數決遮罩
    def __init__(self, tag):
        self.tag = tag
        # kfold_unet_variants.CONFIGS 裡的名稱都是 U-Net(例如 r34_imagenet_aug、r34_imagenet_aug_jpeg),其餘是 YOLO
        self.unet = tag in CONFIGS
        if self.unet:
            self.models = []
            for fold in FOLDS:
                unet = build_model({**CONFIGS[tag], "pretrained": False}).to(DEVICE)
                unet.load_state_dict(torch.load(MASKS / "weights" / f"{tag}_fold{fold}.pth", map_location=DEVICE)["model_state_dict"])
                self.models.append(unet.eval())
        else:
            self.models = [YOLO(yolo_dir(PROJECT, tag, fold) / "weights" / "best.pt") for fold in FOLDS]

    def __call__(self, roi):
        preds = []
        for model in self.models:
            if self.unet:
                preds.append(predict_masks(model, cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), CONFIGS[self.tag]["size"]))
            else:
                result = model.predict(roi, conf=MIN_CONF, iou=IOU, max_det=50, retina_masks=True, verbose=False)[0]
                shape = tuple(result.orig_shape)
                preds.append({cls: items[0][1] if items else np.zeros(shape, dtype=bool) for cls, items in get_candidates(result, shape).items()})
        return {cls: np.sum([p[cls] for p in preds], axis=0) > len(preds) / 2 for cls in NAMES.values()}


def largest_contour(mask):
    if not mask.any():
        return None
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cs, _ = cv2.findContours((labels == largest).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float)


def position_of(mask, offset, lines):
    poly = largest_contour(mask)
    if poly is None or not lines:
        return ""
    # 2026-10-27 使用者選定: 牙根主成分沿長軸延伸的參考線(方法 A,items.json 的 lines_a)+「中央一半」規則
    # (2026-10-25 原為垂直咬合平面的 lines)
    return classify(poly + offset, lines, "center")[0] or ""


def summarize(rows, tags, out):
    summary = []
    for tag in tags:
        sub = [r for r in rows if r["model"] == tag]
        for setting in ("manual", "auto"):
            row = {"model": tag, "setting": setting, "n": len(sub)}
            for k in KEYS:
                values = [r[f"{setting}_{k}"] for r in sub if r[f"{setting}_{k}"] == r[f"{setting}_{k}"]]
                row[k] = round(statistics.mean(values), 4) if values else float("nan")
            both = [r for r in sub if r["rule_gt"] and r[f"{setting}_pos"]]
            row["pos_same_as_gt_mf"] = round(statistics.mean(str(r[f"{setting}_pos"]) == str(r["rule_gt"]) for r in both), 4) if both else float("nan")
            row["pos_n"] = len(both)
            summary.append(row)
        pairs = [(r["manual_mf_dice"], r["auto_mf_dice"]) for r in sub]
        diff = [b - a for a, b in pairs]
        p_dice = wilcoxon([a for a, _ in pairs], [b for _, b in pairs]).pvalue if any(diff) else 1.0
        better = sum(1 for r in sub if not r["manual_mf_hit_1R"] and r["auto_mf_hit_1R"])
        worse = sum(1 for r in sub if r["manual_mf_hit_1R"] and not r["auto_mf_hit_1R"])
        p_hit = binomtest(min(better, worse), better + worse, 0.5).pvalue if better + worse else 1.0
        summary[-1].update(mf_dice_wilcoxon_p=round(p_dice, 4), hit_auto_better=better, hit_auto_worse=worse, hit_mcnemar_p=round(p_hit, 4))
    with open(out / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in summary for k in r)))
        writer.writeheader()
        writer.writerows(summary)
    for r in summary:
        print(" | ".join(f"{k}={v}" for k, v in r.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=[UNET_TAG, "yolo26s-seg_b8", "yolov8n-seg"])
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 張全景片(試跑用)")
    parser.add_argument("--jpeg", action="store_true", help="自動 ROI 照當初的流程存讀兩次 JPEG(品質 95),輸出資料夾名稱加上 _jpeg")
    parser.add_argument("--name", default=OUT.name, help="輸出資料夾名稱(compare_results/ 底下),避免覆蓋其他模型的結果")
    args = parser.parse_args()
    out = OUT.with_name(args.name + ("_jpeg" if args.jpeg else ""))
    out.mkdir(parents=True, exist_ok=True)
    rois = test_rois()
    panos = sorted(rois)[:args.limit or None]
    items = json.loads(ITEMS_JSON.read_text(encoding="utf-8"))["items"]
    with open(KEY_CSV, newline="") as f:
        keys = {r["stem"]: r for r in csv.DictReader(f)}
    print(f"全景片 {len(panos)} 張、測試 ROI {sum(len(rois[p]) for p in panos)} 個")

    teeth = Sized(YOLO(TEETH_MODEL), 1024)
    autos = {}
    for pano in panos:
        img = cv2.imread(str(PANO_DIR / pano))
        autos[pano] = (img, {side: (x1, y1, x2, y2) for side, x1, y1, x2, y2 in auto_rois(img, teeth)})
    del teeth

    rows = []
    with torch.inference_mode():
        for tag in args.models:
            stage2 = Stage2(tag)
            for pano in panos:
                img, auto = autos[pano]
                for stem, side, x, y, w, h in rois[pano]:
                    key = keys.get(stem)
                    lines = lines_from_item(items[key["alias"]], int(key["crop_x"]), int(key["crop_y"]), "a") if key else {}
                    # 人工 ROI: kfold_pano 存好的遮罩多數決
                    shape = cv2.imread(str(GT_IMAGES / f"{stem}.jpg"), cv2.IMREAD_GRAYSCALE).shape
                    gt_manual = load_gt(stem, shape, GT_LABELS, {})
                    preds = [read_masks(MASKS, "test", tag, fold, stem) for fold in FOLDS]
                    vote = {cls: np.sum([p[cls] for p in preds], axis=0) > len(preds) / 2 for cls in NAMES.values()}
                    manual = image_metrics(vote, gt_manual)
                    row = {"model": tag, "pano": pano, "stem": stem, "side": side, "rule_gt": key["auto_position_center"] if key else "",
                           **{f"manual_{k}": manual.get(k, float("nan")) for k in KEYS},
                           "manual_pos": position_of(vote["mf"], np.array([x, y]), lines)}
                    # 自動 ROI
                    polys = gt_polygons(stem, x, y, w, h)
                    mf_pts = np.concatenate(polys["mf"]) if polys["mf"] else None
                    box = auto.get(side)
                    if box is None:
                        row.update(auto_roi=0, mf_covered=0, auto_mf_dice=0.0, auto_mf_hit_1R=0.0, auto_ian_dice=0.0, auto_mf_predicted=0.0, auto_pos="")
                    else:
                        x1, y1, x2, y2 = box
                        crop = img[y1:y2, x1:x2]
                        roi = unsharp_mask(jpeg(crop) if args.jpeg else crop, **UNSHARP)
                        roi = jpeg(roi) if args.jpeg else roi
                        gt_auto = {}
                        for cls in NAMES.values():
                            m = np.zeros(roi.shape[:2], dtype=np.uint8)
                            for poly in polys[cls]:
                                cv2.fillPoly(m, [np.round(poly - (x1, y1)).astype(np.int32)], 1)
                            gt_auto[cls] = m.astype(bool)
                        covered = mf_pts is not None and mf_pts[:, 0].min() >= x1 and mf_pts[:, 1].min() >= y1 and mf_pts[:, 0].max() <= x2 and mf_pts[:, 1].max() <= y2
                        pred = stage2(roi)
                        auto_m = image_metrics(pred, gt_auto) if gt_auto["mf"].any() else {"mf_dice": 0.0, "mf_hit_1R": 0.0, "ian_dice": float("nan"), "mf_predicted": float(pred["mf"].any())}
                        row.update(auto_roi=1, mf_covered=int(covered), **{f"auto_{k}": auto_m.get(k, float("nan")) for k in KEYS},
                                   auto_pos=position_of(pred["mf"], np.array([x1, y1]), lines),
                                   auto_iou_manual=round(max(0, min(x2, x + w) - max(x1, x)) * max(0, min(y2, y + h) - max(y1, y)) /
                                                         ((x2 - x1) * (y2 - y1) + w * h - max(0, min(x2, x + w) - max(x1, x)) * max(0, min(y2, y + h) - max(y1, y))), 4))
                    rows.append(row)
            print(f"{tag}: 完成")
            del stage2
            torch.cuda.empty_cache()
    with open(out / "per_roi.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    summarize(rows, args.models, out)


if __name__ == "__main__":
    main()
