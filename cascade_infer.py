import argparse
import csv
import itertools
import sys

import numpy as np

from compare_yolo_unet import NAMES, ROOT
from kfold_pano import PROJECT
from kfold_yolo_unet import K_FOLDS
from paper_metrics import image_metrics, read_masks
from pano_uncertainty import iou
from pano_visibility import load_test

# 串接式推論(2026-10-27): 五折多數決比單一模型多約 6 點命中,但運算是 5 倍。
#   策略: 先跑 k 折(k = 2 或 3);這 k 個預測的頦孔一致就停,不一致才補跑其餘各折、做五折多數決。
#     一致 = 成員兩兩的頦孔 IoU 都 ≥ tau(兩者都沒預測頦孔算一致);停下時的預測 = 這 k 個的逐像素多數決(k = 2 時為交集)。
#   對每個種子、每一種「先跑哪 k 折」的組合(C(5,k) 種)平均,避免挑到特別好的折;報 測試集命中、mf Dice、平均跑幾折。
#   只讀已存的測試集遮罩,不重新推論、不需要 GPU。CV 每張只有一個模型看過它,不適用。
TAUS = [0.0, 0.3, 0.5, 0.7, 0.9]


def vote(preds):
    return {cls: np.sum([p[cls] for p in preds], axis=0) > len(preds) / 2 for cls in NAMES.values()}


def agree(preds, tau):
    for a, b in itertools.combinations(preds, 2):
        if a["mf"].any() != b["mf"].any():
            return False
        if a["mf"].any() and iou(a["mf"], b["mf"]) < tau:
            return False
    return True


def score(pred, gt, cache, key):
    # 同一個預測遮罩只算一次指標
    if key not in cache:
        m = image_metrics(pred, gt)
        cache[key] = (m["mf_hit_1R"], m["mf_dice"])
    return cache[key]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["r34_imagenet_aug", "yolo26s-seg"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--project", default=PROJECT)
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    out_dir = ROOT / "compare_results" / args.project
    test = [(stem, gt) for stem, gt in load_test() if gt["mf"].any()]  # 沒有 GT 頦孔的不算命中
    folds = list(range(K_FOLDS))

    rows = []
    for model in args.models:
        for seed in args.seeds:
            tag = model if seed == 0 else f"{model}_s{seed}"
            acc = {}  # {(policy, tau): [hit, dice, folds_run] 的逐張列表}
            for stem, gt in test:
                preds = [read_masks(out_dir, "test", tag, f, stem) for f in folds]
                cache = {}
                full = score(vote(preds), gt, cache, "full")
                acc.setdefault(("full", ""), []).append((*full, K_FOLDS))
                for f in folds:
                    acc.setdefault(("single", ""), []).append((*score(preds[f], gt, cache, f"single{f}"), 1))
                for k in (2, 3):
                    for first in itertools.combinations(folds, k):
                        part = [preds[f] for f in first]
                        early = score(vote(part), gt, cache, f"vote{first}")
                        for tau in TAUS:
                            stop = agree(part, tau)
                            acc.setdefault((f"cascade{k}", tau), []).append((*early, k) if stop else (*full, K_FOLDS))
            for (policy, tau), values in acc.items():
                v = np.array(values, float)
                rows.append({"model": model, "seed": seed, "policy": policy, "tau": tau,
                             "hit": round(float(v[:, 0].mean()), 4), "mf_dice": round(float(np.nanmean(v[:, 1])), 4),
                             "folds_run": round(float(v[:, 2].mean()), 3)})
            print(f"{tag} 完成", flush=True)

    path = out_dir / f"cascade_s{''.join(map(str, args.seeds))}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # 跨種子平均
    print(f"\n測試集 {len(test)} 張(有 GT 頦孔),種子 {args.seeds} 平均")
    print(f"{'model':<18}{'policy':<10}{'tau':>5}{'hit':>8}{'mf_dice':>9}{'folds':>7}")
    keys = sorted({(r["model"], r["policy"], str(r["tau"])) for r in rows}, key=lambda k: (k[0], k[1] != "single", k[1] == "full", k[1], k[2]))
    for model, policy, tau in keys:
        sel = [r for r in rows if r["model"] == model and r["policy"] == policy and str(r["tau"]) == tau]
        print(f"{model:<18}{policy:<10}{tau:>5}{np.mean([r['hit'] for r in sel]):>8.4f}"
              f"{np.mean([r['mf_dice'] for r in sel]):>9.4f}{np.mean([r['folds_run'] for r in sel]):>7.2f}")
    print(f"\n輸出: {path}")


if __name__ == "__main__":
    main()
