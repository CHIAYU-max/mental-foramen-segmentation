import argparse
import csv
import statistics
from collections import defaultdict

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from stage2.compare_yolo_unet import NAMES, ROOT, load_gt
from analysis.e2e_eval import FOLDS, GT_IMAGES, GT_LABELS, MASKS, Stage2, gt_polygons, jpeg, test_rois
from stage2.kfold_pano import PROJECT, yolo_dir
from stage2.kfold_patient import UNET_TAG
from stage2.kfold_unet_variants import CONFIGS, build_model
from stage2.mf_postproc import largest, mf_metrics
from data.mf_unsharp_mask import unsharp_mask
from analysis.paper_metrics import SIDE, pixel_metrics, read_masks
from deployment.pipeline_timing import PANO_DIR, TEETH_MODEL, UNSHARP
from stage2.predict_unet import DEVICE
from analysis.seed_bootstrap import cluster_test
from stage1.stage1_eval import Sized, auto_rois

# 自動 ROI 的端到端重算(2026-11-01,使用者要求): e2e_eval.py 只有種子 0、沒有 lcc。流程相同(自動 ROI → unsharp → 五折多數決),
#   多數決後頦孔只留最大區塊(與 mf_postproc.py 相同),五個種子每張取平均,再以病人為組比較(同 slim_eval.py)。
#   條件: manual = kfold_pano 存好的遮罩(控制組,應等於主結果);auto_jpeg = 存讀兩次 JPEG;auto_mem = 記憶體內處理。
#   該側沒有自動 ROI → mf 算沒命中、Dice 0,ian Dice 0。只讀權重與已存遮罩,輸出 compare_results/e2e_seeds/。
OUT = ROOT / "compare_results" / "e2e_seeds"
MODELS = (UNET_TAG, "yolo26s-seg")
CONDITIONS = ("manual", "auto_jpeg", "auto_mem")
METRICS = ("mf_hit_1R", "mf_dice", "ian_dice")


class SeedStage2(Stage2):
    def __init__(self, tag, seed):
        self.tag = tag
        self.unet = tag in CONFIGS
        suffix = f"_s{seed}" if seed else ""
        if self.unet:
            self.models = []
            for fold in FOLDS:
                unet = build_model({**CONFIGS[tag], "pretrained": False}).to(DEVICE)
                unet.load_state_dict(torch.load(MASKS / "weights" / f"{tag}{suffix}_fold{fold}.pth", map_location=DEVICE)["model_state_dict"])
                self.models.append(unet.eval())
        else:
            self.models = [YOLO(yolo_dir(PROJECT, f"{tag}{suffix}", fold) / "weights" / "best.pt") for fold in FOLDS]


def scored(pred, gt):
    return {**mf_metrics(largest(pred["mf"]), gt["mf"]), "ian_dice": pixel_metrics(pred["ian"], gt["ian"])["dice"]}


def auto_row(stage2, img, box, polys, use_jpeg):
    x1, y1, x2, y2 = box
    crop = img[y1:y2, x1:x2]
    roi = unsharp_mask(jpeg(crop) if use_jpeg else crop, **UNSHARP)
    roi = jpeg(roi) if use_jpeg else roi
    gt = {}
    for cls in NAMES.values():
        m = np.zeros(roi.shape[:2], dtype=np.uint8)
        for poly in polys[cls]:
            cv2.fillPoly(m, [np.round(poly - (x1, y1)).astype(np.int32)], 1)
        gt[cls] = m.astype(bool)
    pred = stage2(roi)
    if not gt["mf"].any():
        return {"mf_dice": 0.0, "mf_hit_1R": 0.0, "mf_predicted": float(pred["mf"].any()), "ian_dice": float("nan")}
    return scored(pred, gt)


def compare(rows, base, other):
    # base / other = (模型, 條件);每張對種子取平均,以病人為組
    table = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for side, (model, condition) in (("a", base), ("b", other)):
            if r["model"] == model and r["condition"] == condition:
                table[r["stem"]][side].append(r)
    result = []
    rng = np.random.default_rng(0)
    for key in METRICS:
        pairs = {}
        for stem, v in table.items():
            a, b = [[r[key] for r in v[s] if r[key] == r[key]] for s in "ab"]
            if a and b:
                pairs[stem] = (statistics.mean(a), statistics.mean(b))
        clusters = {}
        for stem, (a, b) in pairs.items():
            c = SIDE.sub("", stem)
            s, n = clusters.get(c, (0.0, 0))
            clusters[c] = (s + b - a, n + 1)
        est, low, high, p = cluster_test(list(clusters.values()), rng)
        row = {"base": "/".join(base), "other": "/".join(other), "metric": key, "base_mean": np.mean([a for a, _ in pairs.values()]),
               "other_mean": np.mean([b for _, b in pairs.values()]), "diff": est, "ci_low": low, "ci_high": high, "p": p, "n": len(pairs),
               "gain": sum(b > a for a, b in pairs.values()), "lost": sum(b < a for a, b in pairs.values())}
        result.append(row)
        print(f"{row['base']} → {row['other']} {key}: {row['base_mean']:.4f} → {row['other_mean']:.4f} 差 {est:+.4f} [{low:+.4f}, {high:+.4f}] "
              f"p {p:.4f} n {row['n']} 得/失 {row['gain']}/{row['lost']}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 張全景片(試跑用)")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rois = test_rois()
    panos = sorted(rois)[:args.limit or None]
    print(f"全景片 {len(panos)} 張、測試 ROI {sum(len(rois[p]) for p in panos)} 個", flush=True)
    teeth = Sized(YOLO(TEETH_MODEL), 1024)
    autos = {}
    for pano in panos:
        img = cv2.imread(str(PANO_DIR / pano))
        autos[pano] = (img, {side: (x1, y1, x2, y2) for side, x1, y1, x2, y2 in auto_rois(img, teeth)})
    del teeth
    missing = sum(side not in autos[p][1] for p in panos for _, side, *_ in rois[p])
    print(f"沒有自動 ROI 的測試 ROI: {missing}", flush=True)

    rows = []
    with torch.inference_mode():
        for tag in MODELS:
            for seed in args.seeds:
                stage2 = SeedStage2(tag, seed)
                masks = f"{tag}_s{seed}" if seed else tag
                for pano in panos:
                    img, auto = autos[pano]
                    for stem, side, x, y, w, h in rois[pano]:
                        shape = cv2.imread(str(GT_IMAGES / f"{stem}.jpg"), cv2.IMREAD_GRAYSCALE).shape
                        preds = [read_masks(MASKS, "test", masks, fold, stem) for fold in FOLDS]
                        vote = {cls: np.sum([p[cls] for p in preds], axis=0) > len(preds) / 2 for cls in NAMES.values()}
                        result = {"manual": scored(vote, load_gt(stem, shape, GT_LABELS, {}))}
                        polys = gt_polygons(stem, x, y, w, h)
                        box = auto.get(side)
                        zero = {"mf_dice": 0.0, "mf_hit_1R": 0.0, "mf_predicted": 0.0, "ian_dice": 0.0}
                        result["auto_jpeg"] = auto_row(stage2, img, box, polys, True) if box else zero
                        result["auto_mem"] = auto_row(stage2, img, box, polys, False) if box else zero
                        for condition, m in result.items():
                            rows.append({"model": tag, "seed": seed, "condition": condition, "stem": stem, **{k: m.get(k, float("nan")) for k in zero}})
                print(f"{tag} 種子 {seed}: 完成", flush=True)
                del stage2
                torch.cuda.empty_cache()
    name = "".join(map(str, args.seeds))
    with open(OUT / f"per_roi_s{name}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for model in MODELS:
        summary += compare(rows, (model, "manual"), (model, "auto_jpeg"))
        summary += compare(rows, (model, "manual"), (model, "auto_mem"))
        summary += compare(rows, (model, "auto_mem"), (model, "auto_jpeg"))
    for condition in CONDITIONS:
        summary += compare(rows, (MODELS[1], condition), (MODELS[0], condition))
    with open(OUT / f"summary_s{name}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
