import csv
import sys

import numpy as np

from compare_yolo_unet import ROOT
from kfold_pano import PROJECT, build_groups
from paper_metrics import SIDE
from pano_uncertainty import auroc

# 2026-10-30 困難案例的信賴區間: 測試集只有 10 個沒命中,「保留 90% 後 92.7% → 97.6%」的不確定性要報出來。
#   資料: pano_uncertainty.py --post lcc 的 uncertainty_per_roi_lcc.csv(不重新推論、CPU)。
#   與 summarize 相同的規則: 依 u 由低到高保留 round(c × n) 張;每次重抽樣都重新排序與保留。
#   95% CI: 以組為單位 bootstrap(CV 用 build_groups、測試用病人,同 seed_bootstrap.py),percentile 法。
#   輸出 compare_results/kfold_pano_mf_559/uncertainty_ci_lcc.csv。
OUT_DIR = ROOT / "compare_results" / PROJECT
N_RESAMPLE = 5000
COVERAGES = (0.95, 0.90, 0.80)


def statistics_of(u, hit):
    order = np.argsort(u, kind="stable")
    out = {"hit@1.00": hit.mean()}
    for c in COVERAGES:
        keep = order[:int(round(c * len(u)))]
        out[f"hit@{c:.2f}"] = hit[keep].mean()
        out[f"gain@{c:.2f}"] = hit[keep].mean() - hit.mean()
    # 全部命中的重抽樣沒有 AUROC 可算
    out["auroc_miss"] = auroc(u, 1 - hit) if 0 < hit.sum() < len(hit) else float("nan")
    return out


def main():
    sys.stdout.reconfigure(line_buffering=True)
    with open(OUT_DIR / "uncertainty_per_roi_lcc.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    results = []
    for setting in ("cv", "test"):
        part = [r for r in rows if r["setting"] == setting]
        stems = [r["stem"] for r in part]
        if setting == "cv":
            groups, _, _, _ = build_groups(stems)
            cluster_of = {s: i for i, members in enumerate(groups) for s in members}
        else:
            cluster_of = {s: SIDE.sub("", s) for s in stems}
        u = np.array([float(r["u"]) for r in part])
        hit = np.array([float(r["hit"]) for r in part])
        clusters = {}
        for index, stem in enumerate(stems):
            clusters.setdefault(cluster_of[stem], []).append(index)
        members = list(clusters.values())
        point = statistics_of(u, hit)
        rng = np.random.default_rng(0)
        samples = {k: [] for k in point}
        for _ in range(N_RESAMPLE):
            picks = rng.integers(0, len(members), len(members))
            index = np.concatenate([members[p] for p in picks])
            for k, v in statistics_of(u[index], hit[index]).items():
                samples[k].append(v)
        print(f"\n===== {setting}: {len(part)} 張,{len(members)} 組,沒命中 {int((hit == 0).sum())} =====")
        for k, value in point.items():
            low, high = np.nanpercentile(samples[k], [2.5, 97.5])
            results.append({"setting": setting, "metric": k, "estimate": round(float(value), 4), "ci_low": round(float(low), 4), "ci_high": round(float(high), 4)})
            print(f"  {k:<12}{value:.4f}  [{low:.4f}, {high:.4f}]")
    with open(OUT_DIR / "uncertainty_ci_lcc.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["setting", "metric", "estimate", "ci_low", "ci_high"])
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
