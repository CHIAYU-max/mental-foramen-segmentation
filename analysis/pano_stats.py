import argparse
import csv
import sys

import cv2
import numpy as np
from scipy.stats import binomtest, wilcoxon

from stage2.compare_yolo_unet import GT_DIR, IMAGES_DIR, NAMES, ROOT, index_files, load_gt
from stage2.kfold_pano import EXCLUDE_TEST, PROJECT
from stage2.kfold_unet_variants import fold_paths
from stage2.kfold_yolo_unet import K_FOLDS
from stage2.mf_postproc import POSTS, apply_post
from analysis.paper_metrics import image_metrics, read_masks

# kfold_pano 的成對檢定(2026-10-26): 以 U-Net 為參考,與各 YOLO 逐張 ROI 比較。
#   只讀 compare_results/kfold_pano_mf_559/masks 下已存好的遮罩,不重新推論、不需要 GPU。
#   CV: 559 張,每張只用沒看過它的那一折;測試: 137 張,五折逐像素多數決(與 kfold_pano.evaluate 相同)。
#   Dice 用 Wilcoxon 符號等級檢定(兩邊都有值的 ROI);命中用精確 McNemar(只看不一致的 ROI)。
#   同一個指標、同一個設定下對多個 YOLO 的 p 值另做 Holm 校正。
#   --post lcc(2026-10-28): 頦孔只留最大連通區塊(CV 每張、測試先多數決再處理),輸出檔名加 _lcc。
#   注意: 同一位病人的多張 ROI 不獨立(CV 尤其明顯),p 值偏樂觀;論文可搭配 paper_metrics.py 的病人層級 bootstrap。
REFERENCE = "r34_imagenet_aug"
OTHERS = ["yolo26s-seg", "yolo11s-seg", "yolo26n-seg", "yolo11n-seg", "yolov8n-seg", "yolov8s-seg"]
DICE_KEYS = ["mf_dice", "ian_dice"]
HIT_KEY = "mf_hit_1R"


def per_roi(tag, out_dir, test, post="none"):
    rows = {"cv": {}, "test": {}}
    for fold in range(K_FOLDS):
        for path in fold_paths(fold, "val", out_dir):
            shape = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).shape
            gt = load_gt(path.stem, shape, path.parent.parent / "labels", {})
            rows["cv"][path.stem] = image_metrics(apply_post(read_masks(out_dir, "cv", tag, fold, path.stem), post), gt)
    for stem, gt in test:
        preds = [read_masks(out_dir, "test", tag, fold, stem) for fold in range(K_FOLDS)]
        vote = {cls: np.sum([p[cls] for p in preds], axis=0) > K_FOLDS / 2 for cls in NAMES.values()}
        rows["test"][stem] = image_metrics(apply_post(vote, post), gt)
    return rows


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def compare(ref, other):
    stems = sorted(set(ref) & set(other))
    result = {"n": len(stems)}
    for key in DICE_KEYS:
        pairs = [(ref[s][key], other[s][key]) for s in stems if ref[s].get(key) == ref[s].get(key) and other[s].get(key) == other[s].get(key)]
        diff = [a - b for a, b in pairs]
        result[f"{key}_n"] = len(pairs)
        result[f"{key}_ref"] = np.mean([a for a, _ in pairs])
        result[f"{key}_other"] = np.mean([b for _, b in pairs])
        result[f"{key}_diff_median"] = float(np.median(diff))
        result[f"{key}_p"] = wilcoxon(diff).pvalue if any(diff) else 1.0
    hits = [(ref[s][HIT_KEY], other[s][HIT_KEY]) for s in stems if HIT_KEY in ref[s] and HIT_KEY in other[s]]
    ref_only = sum(a > b for a, b in hits)
    other_only = sum(b > a for a, b in hits)
    result.update({f"{HIT_KEY}_ref": np.mean([a for a, _ in hits]), f"{HIT_KEY}_other": np.mean([b for _, b in hits]),
                   "hit_ref_only": ref_only, "hit_other_only": other_only,
                   f"{HIT_KEY}_p": binomtest(min(ref_only, other_only), ref_only + other_only, 0.5).pvalue if ref_only + other_only else 1.0})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default=REFERENCE)
    parser.add_argument("--others", nargs="+", default=OTHERS, help="masks 下的模型名稱(例如 yolo26s-seg_b8)")
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--suffix", default="", help="輸出檔名後綴(例如多種子的 _s1),避免覆蓋主結果的 per_roi.csv / stats.csv")
    parser.add_argument("--post", choices=POSTS, default="none", help="頦孔後處理;lcc 時輸出檔名再加 _lcc")
    args = parser.parse_args()
    if args.post != "none":
        args.suffix += f"_{args.post}"
    sys.stdout.reconfigure(line_buffering=True)
    out_dir = ROOT / "compare_results" / args.project

    test_gt = (ROOT / GT_DIR).resolve()
    test = []
    for stem, path in index_files((ROOT / IMAGES_DIR).resolve()).items():
        if stem in EXCLUDE_TEST:
            continue
        shape = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).shape
        test.append((stem, load_gt(stem, shape, test_gt, {})))

    data = {tag: per_roi(tag, out_dir, test, args.post) for tag in [args.reference] + args.others}
    with open(out_dir / f"per_roi{args.suffix}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "setting", "stem", *DICE_KEYS, HIT_KEY])
        for tag, sets in data.items():
            for setting, rows in sets.items():
                for stem, r in sorted(rows.items()):
                    writer.writerow([tag, setting, stem, *(r.get(k, "") for k in DICE_KEYS + [HIT_KEY])])

    results = []
    for setting in ("cv", "test"):
        block = [{"setting": setting, "reference": args.reference, "other": tag, **compare(data[args.reference][setting], data[tag][setting])} for tag in args.others]
        for key in DICE_KEYS + [HIT_KEY]:
            for r, p in zip(block, holm([r[f"{key}_p"] for r in block])):
                r[f"{key}_p_holm"] = p
        results += block
    with open(out_dir / f"stats{args.suffix}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in results)

    print(f"{'setting':<6}{'other':<16}{'n':>5}  {'mf Dice ref/other':<20}{'p':>8}{'Holm':>8}  {'命中 ref/other':<16}{'不一致':>9}{'p':>8}{'Holm':>8}  {'ian Dice ref/other':<20}{'p':>8}")
    for r in results:
        print(f"{r['setting']:<6}{r['other']:<16}{r['n']:>5}  {r['mf_dice_ref']:.4f}/{r['mf_dice_other']:.4f}     {r['mf_dice_p']:>8.4f}{r['mf_dice_p_holm']:>8.4f}  "
              f"{r['mf_hit_1R_ref']:.4f}/{r['mf_hit_1R_other']:.4f} {r['hit_ref_only']:>4}:{r['hit_other_only']:<4}{r['mf_hit_1R_p']:>8.4f}{r['mf_hit_1R_p_holm']:>8.4f}  "
              f"{r['ian_dice_ref']:.4f}/{r['ian_dice_other']:.4f}     {r['ian_dice_p']:>8.4f}")
    print(f"\n逐張結果: {out_dir / f'per_roi{args.suffix}.csv'};檢定: {out_dir / f'stats{args.suffix}.csv'}")


if __name__ == "__main__":
    main()
