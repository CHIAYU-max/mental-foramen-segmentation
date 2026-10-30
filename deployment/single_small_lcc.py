import argparse
import csv
import statistics
import sys

import cv2
import numpy as np
import torch

from stage2.compare_preprocessing import collect_images
from stage2.compare_yolo_unet import GT_DIR, IMAGES_DIR, NAMES, ROOT, index_files, load_gt
from deployment.distill_ensemble import DISTILL_DIR, OUT_DIR, train_student
from stage2.kfold_pano import EXCLUDE_TEST
from stage2.kfold_patient import DATASET
from stage2.kfold_unet_variants import CONFIGS, build_model, predict_masks, to_tensors
from stage2.kfold_yolo_unet import UNET_ARGS, load_samples
from stage2.mf_postproc import apply_post, mf_metrics
from analysis.paper_metrics import SIDE
from stage2.predict_unet import DEVICE
from analysis.seed_bootstrap import cluster_test

# 2026-10-30 小模型的 lcc 版(CV / SOP「1/12 運算」那句用): 蒸餾學生的權重 10/28 已清掉,沒有 lcc 版本,
# 所以重訓單一模型後,用和主結果相同的 lcc 流程評分,不再用「無後處理的單一模型 vs lcc 的五折」比較。
#   模型: 預設 r18_imagenet_aug_384,用全部 559 張訓練 100 epoch、取最後一個 epoch(同 distill_ensemble.py 的 full 條件),種子 0–4。
#   評估: 測試集 137 張的 mf 命中與 Dice(none / lcc);與 r34_imagenet_aug 五折多數決(同種子、lcc)做病人層級配對檢定。
#   輸出: compare_results/kfold_pano_mf_559/distill/single_<config>_lcc_per_roi.csv、single_<config>_lcc_summary.csv。


@torch.inference_mode()
def evaluate(path, config, test):
    model = build_model({**CONFIGS[config], "pretrained": False}).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE)["model_state_dict"])
    model.eval()
    rows = []
    for stem, image, gt in test:
        pred = predict_masks(model, image, CONFIGS[config]["size"])
        for post in ("none", "lcc"):
            rows.append({"stem": stem, "post": post, **mf_metrics(apply_post(pred, post)["mf"], gt["mf"])})
    del model
    return rows


def ensemble_hits(seed):
    # 五折多數決(lcc)的逐張結果,與主結果同一份
    suffix = "" if seed == 0 else f"_s{seed}"
    tag = f"r34_imagenet_aug{suffix}"
    with open(OUT_DIR / f"per_roi{suffix}_lcc.csv", newline="") as f:
        return {r["stem"]: (float(r["mf_hit_1R"]), float(r["mf_dice"])) for r in csv.DictReader(f)
                if r["model"] == tag and r["setting"] == "test"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="r18_imagenet_aug_384", choices=list(CONFIGS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=UNET_ARGS["epochs"])
    parser.add_argument("--tag", default="singlelcc", help="權重檔名的標記;避免覆蓋舊的 full559")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    samples = load_samples(sorted(collect_images((ROOT / DATASET).resolve()).values()))
    x, y = to_tensors(samples, CONFIGS[args.config]["size"])
    # 只用 GT(alpha 0): soft label 不參與 loss,給全零即可沿用 train_student 的增強流程
    soft = torch.zeros(len(x), len(NAMES) + 1, *x.shape[-2:], dtype=torch.float16)
    print(f"訓練 ROI: {len(x)} 張,設定 {args.config},種子 {args.seeds}")

    test = []
    for stem, path in sorted(index_files((ROOT / IMAGES_DIR).resolve()).items()):
        if stem in EXCLUDE_TEST:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        test.append((stem, image, load_gt(stem, image.shape, (ROOT / GT_DIR).resolve(), {})))
    print(f"測試 ROI: {len(test)} 張")

    per_roi = []
    for seed in args.seeds:
        path = train_student(f"{args.config}_{args.tag}_s{seed}", args.config, x, y, soft, 0.0, args.epochs, seed)
        rows = evaluate(path, args.config, test)
        per_roi += [{"seed": seed, **r} for r in rows]
        for post in ("none", "lcc"):
            part = [r for r in rows if r["post"] == post]
            print(f"種子 {seed} {post}: 命中 {np.mean([r['mf_hit_1R'] for r in part]):.4f}  mf Dice {np.nanmean([r['mf_dice'] for r in part]):.4f}")

    DISTILL_DIR.mkdir(parents=True, exist_ok=True)
    with open(DISTILL_DIR / f"single_{args.config}_lcc_per_roi.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "stem", "post", "mf_dice", "mf_predicted", "mf_hit_1R"])
        writer.writeheader()
        writer.writerows(per_roi)

    summary = []
    for post in ("none", "lcc"):
        hits = [np.mean([r["mf_hit_1R"] for r in per_roi if r["seed"] == s and r["post"] == post]) for s in args.seeds]
        dice = [np.nanmean([r["mf_dice"] for r in per_roi if r["seed"] == s and r["post"] == post]) for s in args.seeds]
        sd = statistics.stdev(hits) if len(hits) > 1 else 0.0
        summary.append({"row": f"{args.config} single {post}", "hit": round(float(np.mean(hits)), 4), "hit_sd": round(sd, 4), "mf_dice": round(float(np.mean(dice)), 4)})

    ensemble = {s: ensemble_hits(s) for s in args.seeds}
    hits = [np.mean([v[0] for v in ensemble[s].values()]) for s in args.seeds]
    dice = [np.nanmean([v[1] for v in ensemble[s].values()]) for s in args.seeds]
    summary.append({"row": "r34_imagenet_aug ensemble lcc", "hit": round(float(np.mean(hits)), 4),
                    "hit_sd": round(statistics.stdev(hits) if len(hits) > 1 else 0.0, 4), "mf_dice": round(float(np.mean(dice)), 4)})

    # 病人層級配對檢定: 每張 ROI 先對種子取平均,再算 五折 − 單一模型
    rng = np.random.default_rng(0)
    for index, metric in enumerate(("mf_hit_1R", "mf_dice")):
        clusters = {}
        for stem in {r["stem"] for r in per_roi}:
            single = [r[metric] for r in per_roi if r["stem"] == stem and r["post"] == "lcc"]
            full = [ensemble[s][stem][index] for s in args.seeds]
            if any(v != v for v in single + full):
                continue
            key = SIDE.sub("", stem)
            total, n = clusters.get(key, (0.0, 0))
            clusters[key] = (total + float(np.mean(full) - np.mean(single)), n + 1)
        est, low, high, p = cluster_test(list(clusters.values()), rng)
        summary.append({"row": f"ensemble - single {metric}", "hit": round(est, 4), "ci_low": round(low, 4), "ci_high": round(high, 4), "p": round(p, 4)})

    with open(DISTILL_DIR / f"single_{args.config}_lcc_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["row", "hit", "hit_sd", "mf_dice", "ci_low", "ci_high", "p"])
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
