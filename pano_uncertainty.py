import argparse
import csv
import sys

import cv2
import numpy as np

from compare_yolo_unet import NAMES, ROOT, load_gt
from kfold_pano import PROJECT
from kfold_unet_variants import fold_paths
from kfold_yolo_unet import K_FOLDS
from paper_metrics import image_metrics, read_masks
from pano_visibility import VISIBILITY_CSV, load_test

# 用模型之間的分歧標出「需要醫師確認」的 ROI(2026-10-26):
#   瓶頸在頦孔本身看不清楚,不在模型;與其再提高準確率,不如讓系統知道自己哪裡沒把握。
#   成員: CV 為同一折的多個訓練種子(每張 3 個預測);測試為 種子 × 5 折(每張 15 個預測)。
#   預測 = 成員的逐像素多數決;分歧度 u = 1 - 平均 IoU(成員的頦孔遮罩, 多數決遮罩),
#   成員或多數決沒有頦孔時該成員的 IoU 記為 0(多數決與成員都沒有則記為 1)。不需要 GT,部署時也算得出來。
#   評估: AUROC(u 預測「沒命中」/ 預測「無法辨識」)、依 u 由低到高保留 ROI 時的命中率與 Dice(risk-coverage)。
#   只讀已存的遮罩,不重新推論、不需要 GPU。
#   --post lcc(2026-10-29): 多數決後頦孔只留最大區塊再評分(u 仍用原本的成員遮罩),輸出檔名加 _lcc。
SEEDS = ["r34_imagenet_aug", "r34_imagenet_aug_s1", "r34_imagenet_aug_s2"]
COVERAGES = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7]


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / union if union else 1.0


def disagreement(preds):
    vote = {cls: np.sum([p[cls] for p in preds], axis=0) > len(preds) / 2 for cls in NAMES.values()}
    if not vote["mf"].any() and not any(p["mf"].any() for p in preds):
        return vote, 0.0
    return vote, 1.0 - float(np.mean([iou(p["mf"], vote["mf"]) if p["mf"].any() and vote["mf"].any() else 0.0 for p in preds]))


def auroc(scores, labels):
    # labels 1 = 正類(沒命中 / 無法辨識);以等級和計算,同分取平均等級
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = labels.sum(), len(labels) - labels.sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    for value in np.unique(scores):
        tie = scores == value
        ranks[tie] = ranks[tie].mean()
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def summarize(setting, rows, visible):
    u = np.array([r["u"] for r in rows])
    hit = np.array([r["hit"] for r in rows])
    dice = np.array([r["dice"] for r in rows])
    vis = [(r["u"], 1 - visible[r["stem"]]) for r in rows if r["stem"] in visible]
    out = {"setting": setting, "n": len(rows), "misses": int((hit == 0).sum()),
           "auroc_miss": auroc(u, 1 - hit), "auroc_not_visible": auroc(*zip(*vis)) if vis else float("nan")}
    order = np.argsort(u, kind="stable")
    for c in COVERAGES:
        keep = order[:int(round(c * len(rows)))]
        out[f"hit@{c:.2f}"] = float(hit[keep].mean())
        out[f"dice@{c:.2f}"] = float(np.nanmean(dice[keep]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", default=SEEDS, help="同一設定、不同訓練種子的 masks 名稱")
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--post", choices=["none", "lcc"], default="none")
    args = parser.parse_args()
    from mf_postproc import apply_post
    suffix = "" if args.post == "none" else f"_{args.post}"
    sys.stdout.reconfigure(line_buffering=True)
    out_dir = ROOT / "compare_results" / args.project
    with open(VISIBILITY_CSV, newline="") as f:
        visible = {r["stem"]: int(r["mf_visible"]) for r in csv.DictReader(f)}

    per_roi = []
    for fold in range(K_FOLDS):
        for path in fold_paths(fold, "val", out_dir):
            shape = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).shape
            gt = load_gt(path.stem, shape, path.parent.parent / "labels", {})
            vote, u = disagreement([read_masks(out_dir, "cv", tag, fold, path.stem) for tag in args.seeds])
            m = image_metrics(apply_post(vote, args.post), gt)
            per_roi.append({"setting": "cv", "stem": path.stem, "members": len(args.seeds), "u": u,
                            "hit": m.get("mf_hit_1R", float("nan")), "dice": m["mf_dice"]})
    for stem, gt in load_test():
        vote, u = disagreement([read_masks(out_dir, "test", tag, fold, stem) for tag in args.seeds for fold in range(K_FOLDS)])
        m = image_metrics(apply_post(vote, args.post), gt)
        per_roi.append({"setting": "test", "stem": stem, "members": len(args.seeds) * K_FOLDS, "u": u,
                        "hit": m.get("mf_hit_1R", float("nan")), "dice": m["mf_dice"]})
    # 沒有 GT 頦孔的 ROI 沒有命中與否,不納入
    per_roi = [r for r in per_roi if r["hit"] == r["hit"]]

    summary = [summarize(s, [r for r in per_roi if r["setting"] == s], visible) for s in ("cv", "test")]
    for s in summary:
        print(f"\n===== {s['setting']}:{s['n']} 張,多數決沒命中 {s['misses']} 張 =====")
        print(f"  AUROC(分歧度 → 沒命中) {s['auroc_miss']:.3f};AUROC(分歧度 → 無法辨識) {s['auroc_not_visible']:.3f}")
        print("  依分歧度保留 ROI(其餘交給醫師確認):")
        for c in COVERAGES:
            print(f"    保留 {c:>4.0%}: 命中 {s[f'hit@{c:.2f}']:.1%}   mf Dice {s[f'dice@{c:.2f}']:.3f}")

    with open(out_dir / f"uncertainty_per_roi{suffix}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_roi[0]) + ["mf_visible"])
        writer.writeheader()
        for r in per_roi:
            writer.writerow({**{k: round(v, 5) if isinstance(v, float) else v for k, v in r.items()}, "mf_visible": visible.get(r["stem"], "")})
    with open(out_dir / f"uncertainty_summary{suffix}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows({k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()} for s in summary)

    # risk-coverage 圖(論文用)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for s, style in (("cv", "-"), ("test", "--")):
        rows = sorted((r for r in per_roi if r["setting"] == s), key=lambda r: r["u"])
        hits = np.cumsum([r["hit"] for r in rows]) / np.arange(1, len(rows) + 1)
        cover = np.arange(1, len(rows) + 1) / len(rows)
        ax.plot(cover * 100, hits * 100, style, label=f"{s} (n={len(rows)})")
    ax.set_xlabel("Coverage: ROIs kept (%)")
    ax.set_ylabel("Mental foramen hit rate (%)")
    ax.set_xlim(50, 100)
    ax.set_ylim(75, 100)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"risk_coverage{suffix}.png", dpi=200)
    print(f"\n輸出: {out_dir / f'uncertainty_per_roi{suffix}.csv'}、uncertainty_summary{suffix}.csv、risk_coverage{suffix}.png")


if __name__ == "__main__":
    main()
