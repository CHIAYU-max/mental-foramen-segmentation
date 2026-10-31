import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from stage2.compare_yolo_unet import ROOT
from analysis.e2e_eval import jpeg
from stage1.fdi_cropped import assign_fdi_indices, find_mandibular_border, get_split_roi_bounds, process_and_separate_jaws
from stage2.kfold_pano import PROJECT
from stage2.kfold_patient import save_masks
from stage2.kfold_unet_variants import CONFIGS, build_model, predict_masks
from stage2.kfold_yolo_unet import K_FOLDS
from stage2.mf_postproc import largest
from data.mf_unsharp_mask import unsharp_mask
from analysis.pano_uncertainty import disagreement
from analysis.paper_metrics import read_masks
from deployment.pipeline_timing import TEETH_MODEL, UNSHARP
from stage2.predict_unet import DEVICE
from stage1.stage1_eval import Sized

# 外部資料 Tufts Dental Database(2026-10-28,Kaggle 第三方鏡像,授權未確認,只做內部實驗):
#   1000 張全景片(840 × 1615,比我們的 1280 × 2648 小約 0.63 倍),逐顆牙齒多邊形(Universal 編號 1–32,乳牙 A–T),沒有頦孔 / ian。
#   A. 第一階段(不需要新標註): 牙齒模型(raw、imgsz 1024)→ FDI 中線修正版 → 裁 ROI,與目前流程相同。
#      - 每側有沒有 ROI、失敗原因
#      - 牙齒 GT 評估 FDI 編號: 預測的 x4 / x5 / x6 質心落在哪一顆 GT 牙齒的多邊形內(Universal 換成 FDI),編號相同才算對。
#        乳牙 / 混合齒列的全景片另外列(第一階段沒有為兒童設計)。
#   B. 第二階段(不需要新標註): 自動 ROI 照當初的流程(存讀 JPEG → unsharp mask → 存讀 JPEG)→ R34/512 種子 0–4 × 五折,
#      遮罩存到 masks/tufts/<設定>/fold<k>/;每個 ROI 算多數決有沒有頦孔(lcc)與分歧度 u,
#      與我們的三層可辨識度(可辨識 / 無法辨識 / 沒有標註)比較(各層的數字見 EXPERIMENTS.md「沒有標註的 ROI」)。
#   輸出 compare_results/tufts/(sides.csv、fdi.csv、rois.csv、summary.txt);ROI 影像存在 external/tufts/rois/(之後挑來標頦孔)。
#   --scale(2026-10-28): 先把全景片放大(1.58 ≈ 對齊我們的像素大小 1280 / 840),牙齒 GT 一起放大;輸出與 ROI 資料夾加 --name。
#   --members folds: 只有種子 0 的機器(筆電)改用同一個種子的 5 折當成員算 u(與 5 種子版不能直接比,只在同一設定內比較放大前後)。
TUFTS = ROOT / "external" / "tufts"
OUT = ROOT / "compare_results" / "tufts"
TAGS = ["r34_imagenet_aug"] + [f"r34_imagenet_aug_s{s}" for s in range(1, 5)]
TARGETS = {"34", "35", "36", "44", "45", "46"}


def universal_to_fdi(title):
    # 下顎恆牙: Universal 17–24 = FDI 38–31(病人左側),25–32 = FDI 41–48(病人右側);其餘(上顎、乳牙)回傳 None
    if not title.isdigit():
        return None
    u = int(title)
    if 17 <= u <= 24:
        return f"3{25 - u}"
    if 25 <= u <= 32:
        return f"4{u - 24}"
    return None


def gt_teeth(entry, shape, scale=1.0):
    # {FDI: 遮罩}(只有下顎恆牙),以及這張片子有沒有乳牙;scale = 全景片放大倍率
    teeth, primary = {}, False
    for o in entry["Label"]["objects"]:
        if not o["title"].isdigit():
            primary = primary or bool(re.fullmatch(r"[A-T]", o["title"]))
            continue
        fdi = universal_to_fdi(o["title"])
        if fdi is None or o.get("polygons") in (None, "none"):
            continue
        polys = json.loads(o["polygons"]) if isinstance(o["polygons"], str) else o["polygons"]
        mask = np.zeros(shape, dtype=np.uint8)
        for poly in polys:
            if len(poly) >= 3:
                cv2.fillPoly(mask, [np.round(np.array(poly, dtype=np.float32) * scale).astype(np.int32)], 1)
        teeth[fdi] = mask.astype(bool)
    return teeth, primary


def stage1(img, model):
    # 與 stage1_eval.auto_rois 相同,但同時回傳下顎牙齒(含 fdi_id)
    _, lower = process_and_separate_jaws(img, model)
    if not lower:
        return [], {}
    lower = assign_fdi_indices(lower)
    h, w = img.shape[:2]
    rois = {}
    for side, b in get_split_roi_bounds(lower).items():
        jaw_y = min(find_mandibular_border(img, b["y_max_root"], b["x_min"], b["x_max"], b["dynamic_fallback"]), h)
        x2 = min(b["x_max"], w)
        if jaw_y > b["y_upper"] and x2 > b["x_min"]:
            rois[side] = (b["x_min"], b["y_upper"], x2, jaw_y)
    return lower, rois


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 張(試跑用)")
    parser.add_argument("--scale", type=float, default=1.0, help="全景片放大倍率")
    parser.add_argument("--tags", nargs="+", default=TAGS)
    parser.add_argument("--members", choices=["seeds", "folds"], default="seeds", help="u 的成員: 同一折的各種子,或同一種子的各折")
    parser.add_argument("--name", default="", help="輸出資料夾後綴(例如 _x1.58)")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    out_res, roi_dir, mask_set = OUT.with_name(OUT.name + args.name), TUFTS / f"rois{args.name}", "tufts" + args.name
    out_res.mkdir(parents=True, exist_ok=True)
    roi_dir.mkdir(exist_ok=True)
    entries = {Path(e["External ID"]).stem: e for e in json.loads((TUFTS / "teeth_polygon.json").read_text())}
    paths = sorted(TUFTS.joinpath("radiographs").glob("*"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)[:args.limit or None]

    # A. 第一階段
    teeth_model = Sized(YOLO(TEETH_MODEL), 1024)
    sides, fdi_rows, rois = [], [], []
    for path in paths:
        img = cv2.imread(str(path))
        if args.scale != 1.0:
            img = cv2.resize(img, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_CUBIC)
        lower, boxes = stage1(img, teeth_model)
        gt, primary = gt_teeth(entries[path.stem], img.shape[:2], args.scale) if path.stem in entries else ({}, False)
        for side, quad in (("Right", "4"), ("Left", "3")):
            pred = {t["fdi_id"]: t for t in lower if t.get("fdi_id", "")[:1] == quad}
            gt_targets = sorted(k for k in gt if k in TARGETS and k[0] == quad)
            sides.append({"pano": path.stem, "side": side, "primary": int(primary), "n_lower_pred": len(lower), "n_side_pred": len(pred),
                          "gt_targets": " ".join(gt_targets), "roi": int(side in boxes)})
            for fdi in sorted(TARGETS):
                if fdi[0] != quad or fdi not in pred:
                    continue
                t = pred[fdi]
                cx, cy = int(round(t["cx"])), int(round(t["cy"]))
                hit = [k for k, m in gt.items() if 0 <= cy < m.shape[0] and 0 <= cx < m.shape[1] and m[cy, cx]]
                fdi_rows.append({"pano": path.stem, "side": side, "primary": int(primary), "pred_fdi": fdi,
                                 "gt_fdi": hit[0] if hit else "", "correct": int(bool(hit) and hit[0] == fdi)})
            if side in boxes:
                x1, y1, x2, y2 = boxes[side]
                roi = jpeg(unsharp_mask(jpeg(img[y1:y2, x1:x2]), **UNSHARP))
                stem = f"{path.stem}_{side}_roi"
                cv2.imwrite(str(roi_dir / f"{stem}.png"), roi)  # PNG: 已經模擬過兩次 JPEG,不要再壓一次
                rois.append({"stem": stem, "pano": path.stem, "side": side, "primary": int(primary), "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    del teeth_model
    torch.cuda.empty_cache()
    print(f"第一階段完成: 全景片 {len(paths)} 張、ROI {len(rois)} 個")

    # B. 第二階段
    out_dir = ROOT / "compare_results" / PROJECT
    for tag in args.tags:
        config = CONFIGS[re.sub(r"_s\d+$", "", tag)]
        for fold in range(K_FOLDS):
            unet = build_model({**config, "pretrained": False}).to(DEVICE)
            unet.load_state_dict(torch.load(out_dir / "weights" / f"{tag}_fold{fold}.pth", map_location=DEVICE)["model_state_dict"])
            unet.eval()
            for r in rois:
                image = cv2.imread(str(roi_dir / f"{r['stem']}.png"), cv2.IMREAD_GRAYSCALE)
                save_masks(predict_masks(unet, image, config["size"]), out_dir / "masks" / mask_set / tag / f"fold{fold}" / f"{r['stem']}.png")
            del unet
            torch.cuda.empty_cache()
        print(f"[完成] {tag}")
    # 成員組: seeds = 每一折的各種子(5 組,各組 len(tags) 個成員);folds = 每個種子的 5 折(len(tags) 組,各組 5 個成員)
    groups = ([[(t, f) for t in args.tags] for f in range(K_FOLDS)] if args.members == "seeds"
              else [[(t, f) for f in range(K_FOLDS)] for t in args.tags])
    for r in rois:
        us, present = [], []
        for group in groups:
            vote, u = disagreement([read_masks(out_dir, mask_set, t, f, r["stem"]) for t, f in group])
            us.append(u)
            present.append(int(largest(vote["mf"]).any()))
        r.update(u=round(statistics.mean(us), 4), mf_present=round(statistics.mean(present), 2))

    for name, rows in (("sides.csv", sides), ("fdi.csv", fdi_rows), ("rois.csv", rois)):
        with open(out_res / name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            writer.writeheader()
            writer.writerows(rows)

    lines = []
    for label, keep in (("恆牙", 0), ("有乳牙", 1)):
        s = [r for r in sides if r["primary"] == keep]
        f = [r for r in fdi_rows if r["primary"] == keep]
        ro = [r for r in rois if r["primary"] == keep]
        if not s:
            continue
        lines.append(f"[{label}] 全景片 {len(s) // 2} 張、側 {len(s)}: 有 ROI {sum(r['roi'] for r in s)}({sum(r['roi'] for r in s) / len(s):.1%})")
        if f:
            lines.append(f"  x4–x6 編號正確 {sum(r['correct'] for r in f)} / {len(f)}({sum(r['correct'] for r in f) / len(f):.1%});"
                         f"質心不在任何 GT 牙齒內 {sum(not r['gt_fdi'] for r in f)};錯誤方向 "
                         + ", ".join(f"{a}→{b} ×{c}" for (a, b), c in Counter((r['gt_fdi'], r['pred_fdi']) for r in f if r['gt_fdi'] and not r['correct']).most_common(6)))
        if ro:
            u = [r["u"] for r in ro]
            lines.append(f"  ROI {len(ro)} 個: 多數決有頦孔 {statistics.mean(r['mf_present'] for r in ro):.3f}、u 中位數 {statistics.median(u):.3f}、平均 {statistics.mean(u):.3f}、"
                         f"u ≤ 0.1(約我們可辨識層的中位數){sum(x <= 0.1 for x in u) / len(u):.1%}")
    lines.insert(0, f"放大 {args.scale}、成員 {args.members}、模型 {' '.join(args.tags)}")
    lines.append("對照(我們的資料,5 種子 × 同一折): 可辨識 u 中位數 0.088 / 有頦孔 1.00;無法辨識 0.167 / 0.95;沒有標註 0.479 / 0.61")
    (out_res / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
