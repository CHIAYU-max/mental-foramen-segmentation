"""2026-11-14 精確度 × 運算量: 串接式策略的比較(使用者要求;只讀已存的測試集遮罩,CPU)。

問題: (1) 有分歧時加跑更多模型(同一架構 5 種子 × 5 折 = 25 個)能不能提高命中? (2) 先用便宜的模型(瘦身解碼器、R34/384),
有分歧才換 R34/512,能不能在少一點運算下保住精確度?
固定(不調): 先跑 2 折、頦孔 IoU >= 0.5(或都沒有頦孔)就停、停下時取聯集;最後輸出 lcc。對 5 個起始種子 × 10 種「先跑哪 2 折」平均。
運算量: 第二階段每個 ROI 的 GMAC(R34/512 40.9、瘦身解碼器 24.13、R34/384 23.0)。遮罩是 FP32(INT8 只有 R34/512 有)。
限制: 在測試集上比較多個策略,屬探索性;CV 每張只有一個折的模型看過,無法模擬。差 = 策略 − 現行(r34 c2→5),以病人為組的 bootstrap。
輸出: compare_results/kfold_pano_mf_559/policy_pareto.csv、policy_pareto_per_roi.csv
"""
import csv
import itertools
from collections import defaultdict

import numpy as np

from deployment.cascade_infer import agree, vote
from stage2.compare_yolo_unet import NAMES, ROOT
from analysis.e2e_seeds import SIDE
from stage2.kfold_pano import PROJECT
from stage2.mf_postproc import apply_post
from analysis.paper_metrics import image_metrics, read_masks
from analysis.pano_visibility import load_test
from analysis.seed_bootstrap import cluster_test

FAMILIES = {"r34": ("r34_imagenet_aug", 40.9), "slim": ("r34_imagenet_aug_slimhalf", 24.13), "r384": ("r34_imagenet_aug_384", 23.0)}
SEEDS, FOLDS, TAU = range(5), range(5), 0.5
PAIRS = list(itertools.combinations(FOLDS, 2))
REF = "r34 c2→5"


def union(a, b):
    return {cls: a[cls] | b[cls] for cls in NAMES.values()}


def main():
    out = ROOT / "compare_results" / PROJECT
    test = [(stem, gt) for stem, gt in load_test() if gt["mf"].any()]
    per = defaultdict(lambda: defaultdict(list))  # policy → stem → [(hit, mf_dice, ian_dice, gmac)]
    for n, (stem, gt) in enumerate(test, 1):
        preds = {f: {s: [read_masks(out, "test", tag if s == 0 else f"{tag}_s{s}", k, stem) for k in FOLDS] for s in SEEDS}
                 for f, (tag, _) in FAMILIES.items()}

        def metric(pred):
            m = image_metrics(apply_post(pred, "lcc"), gt)
            return m["mf_hit_1R"], m["mf_dice"], m["ian_dice"]

        full5 = {f: {s: metric(vote(preds[f][s])) for s in SEEDS} for f in FAMILIES}
        full25 = {f: metric(vote([p for s in SEEDS for p in preds[f][s]])) for f in FAMILIES}
        for f, (_, g) in FAMILIES.items():
            per[f"{f} full25"][stem].append((*full25[f], 25 * g))
            for s in SEEDS:
                per[f"{f} full5"][stem].append((*full5[f][s], 5 * g))
                for k in FOLDS:
                    per[f"{f} single"][stem].append((*metric(preds[f][s][k]), g))
                for a, b in PAIRS:
                    pair = [preds[f][s][a], preds[f][s][b]]
                    stop = agree(pair, TAU)
                    early = metric(union(*pair)) if stop else None
                    per[f"{f} c2→5"][stem].append((*early, 2 * g) if stop else (*full5[f][s], 5 * g))
                    per[f"{f} c2→25"][stem].append((*early, 2 * g) if stop else (*full25[f], 25 * g))
                    if f != "r34":  # 便宜的先跑 2 折,有分歧改跑 R34/512
                        per[f"{f} c2→r34 full5"][stem].append((*early, 2 * g) if stop else (*full5["r34"][s], 2 * g + 5 * 40.9))
                        per[f"{f} c2→r34 full25"][stem].append((*early, 2 * g) if stop else (*full25["r34"], 2 * g + 25 * 40.9))
        if n % 20 == 0:
            print(f"{n}/{len(test)}", flush=True)
    mean = {p: {stem: np.nanmean(np.array(v, float), axis=0) for stem, v in d.items()} for p, d in per.items()}
    rng = np.random.default_rng(0)
    rows = []
    for p, d in mean.items():
        row = {"policy": p, "gmac": np.mean([v[3] for v in d.values()])}
        for i, key in enumerate(("mf_hit_1R", "mf_dice", "ian_dice")):
            row[key] = np.nanmean([v[i] for v in d.values()])
            clusters = {}
            for stem, v in d.items():
                diff = v[i] - mean[REF][stem][i]
                if diff == diff:
                    c = SIDE.sub("", stem)
                    s0, n0 = clusters.get(c, (0.0, 0))
                    clusters[c] = (s0 + diff, n0 + 1)
            est, lo, hi, pv = cluster_test(list(clusters.values()), rng)
            row.update({f"{key}_diff": est, f"{key}_lo": lo, f"{key}_hi": hi, f"{key}_p": pv})
        rows.append(row)
    rows.sort(key=lambda r: r["gmac"])
    with open(out / "policy_pareto.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / "policy_pareto_per_roi.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "stem", "mf_hit_1R", "mf_dice", "ian_dice", "gmac"])
        for p, d in mean.items():
            for stem, v in d.items():
                w.writerow([p, stem, *np.round(v, 5)])
    print(f"{'policy':<24}{'GMAC':>7}{'hit':>8}{'Δhit [95% CI]':>26}{'mfDice':>8}{'Δ':>8}{'ianDice':>9}{'Δ':>8}")
    for r in rows:
        print(f"{r['policy']:<24}{r['gmac']:>7.1f}{r['mf_hit_1R']:>8.4f}  {100 * r['mf_hit_1R_diff']:+6.2f} [{100 * r['mf_hit_1R_lo']:+.2f}, {100 * r['mf_hit_1R_hi']:+.2f}] p {r['mf_hit_1R_p']:.3f}"
              f"{r['mf_dice']:>8.4f}{r['mf_dice_diff']:>+8.4f}{r['ian_dice']:>9.4f}{r['ian_dice_diff']:>+8.4f}")


if __name__ == "__main__":
    main()
