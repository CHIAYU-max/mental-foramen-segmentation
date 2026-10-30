import argparse
import csv
import sys

import numpy as np

from compare_yolo_unet import ROOT
from kfold_pano import PROJECT, build_groups
from paper_metrics import SIDE

# 多種子合併的病人層級檢定(2026-10-26): pano_stats.py 是逐種子、逐張 ROI 的檢定(同一病人的 ROI 不獨立,p 偏樂觀),
# 這裡把 --seeds 個訓練種子合併,並以「組」為單位重抽樣。
#   資料: pano_stats.py 存的逐張結果 per_roi.csv(種子 0)、per_roi_s<k>.csv(種子 k),不重新推論、不需要 GPU。
#   --post lcc(2026-10-28): 讀 pano_stats.py --post lcc 的 per_roi*_lcc.csv,輸出檔名加 _lcc。
#   --prefix(2026-10-28): 讀 per_roi<prefix>[_s<k>][_lcc].csv,用於主結果以外的比較(例如 R34/384 vs R34/512 的 _r384)。
#   每張 ROI 先對種子取平均(命中變成 0–1 之間的比例),再算 參考 − 對手 的差。
#   組: CV 用 kfold_pano.build_groups(同一病人 ∪ 同一來源全景片,與切分一致);測試集用病人(左右側為同一人,同 paper_metrics)。
#   95% CI: 以組為單位 bootstrap 平均差;p: 以組為單位的符號翻轉置換檢定(各組差的總和隨機變號),雙尾。
N_RESAMPLE = 20000
KEYS = ["mf_hit_1R", "mf_dice", "ian_dice"]


def load(seeds, reference, other, out_dir, post="", prefix=""):
    # 回傳 {setting: {stem: {key: [各種子的 參考 − 對手]}}}
    diffs = {}
    for seed in seeds:
        suffix = "" if seed == 0 else f"_s{seed}"
        tags = {reference + suffix: "ref", other + suffix: "other"}
        values = {}
        with open(out_dir / f"per_roi{prefix}{suffix}{post}.csv", newline="") as f:
            for r in csv.DictReader(f):
                if r["model"] in tags:
                    values.setdefault((r["setting"], r["stem"]), {})[tags[r["model"]]] = r
        if not any("ref" in v and "other" in v for v in values.values()):
            raise SystemExit(f"per_roi{prefix}{suffix}{post}.csv 裡沒有 {reference + suffix} 與 {other + suffix},請先跑 pano_stats.py")
        for (setting, stem), pair in values.items():
            if len(pair) < 2:
                continue
            for key in KEYS:
                a, b = pair["ref"][key], pair["other"][key]
                if a != "" and b != "":
                    diffs.setdefault(setting, {}).setdefault(stem, {}).setdefault(key, []).append(float(a) - float(b))
    return diffs


def cluster_test(per_cluster, rng):
    # per_cluster: [(差的總和, 張數)];回傳 (平均差, CI 下界, CI 上界, p)
    sums = np.array([s for s, _ in per_cluster])
    counts = np.array([n for _, n in per_cluster])
    estimate = sums.sum() / counts.sum()
    picks = rng.integers(0, len(sums), (N_RESAMPLE, len(sums)))
    low, high = np.percentile(sums[picks].sum(1) / counts[picks].sum(1), [2.5, 97.5])
    signs = rng.choice([-1, 1], (N_RESAMPLE, len(sums)))
    null = (signs * sums).sum(1) / counts.sum()
    p = (np.sum(np.abs(null) >= abs(estimate)) + 1) / (N_RESAMPLE + 1)
    return estimate, low, high, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="r34_imagenet_aug")
    parser.add_argument("--other", default="yolo26s-seg")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--post", choices=["none", "lcc"], default="none", help="頦孔後處理(讀 pano_stats.py --post 的結果)")
    parser.add_argument("--prefix", default="", help="per_roi 檔名的前綴(pano_stats.py 的 --suffix 去掉 _s<k>)")
    args = parser.parse_args()
    post = "" if args.post == "none" else f"_{args.post}"
    sys.stdout.reconfigure(line_buffering=True)
    out_dir = ROOT / "compare_results" / args.project
    diffs = load(args.seeds, args.reference, args.other, out_dir, post, args.prefix)

    groups, _, _, _ = build_groups(list(diffs["cv"]))
    cluster_of = {"cv": {s: i for i, members in enumerate(groups) for s in members},
                  "test": {s: SIDE.sub("", s) for s in diffs["test"]}}
    rng = np.random.default_rng(0)
    results = []
    # 合併種子,以及逐種子(對照 pano_stats.py)
    for label, pick in [("pooled", None)] + [(f"seed{s}", i) for i, s in enumerate(args.seeds)]:
        for setting in ("cv", "test"):
            for key in KEYS:
                clusters = {}
                for stem, by_key in diffs[setting].items():
                    values = by_key.get(key, [])
                    if len(values) != len(args.seeds):
                        continue  # 有種子缺值(例如沒有 GT)就整張不算,各種子才是同一批 ROI
                    d = float(np.mean(values)) if pick is None else values[pick]
                    s, n = clusters.get(cluster_of[setting][stem], (0.0, 0))
                    clusters[cluster_of[setting][stem]] = (s + d, n + 1)
                est, low, high, p = cluster_test(list(clusters.values()), rng)
                results.append({"seeds": label, "setting": setting, "metric": key, "n_roi": sum(n for _, n in clusters.values()),
                                "n_clusters": len(clusters), "diff": round(est, 4), "ci_low": round(low, 4), "ci_high": round(high, 4), "p": round(p, 4)})
    path = out_dir / f"seed_bootstrap_{args.other}_s{''.join(map(str, args.seeds))}{post}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"{args.reference} − {args.other},種子 {args.seeds},組為單位重抽樣 {N_RESAMPLE} 次")
    for r in results:
        print(f"{r['seeds']:<7}{r['setting']:<5}{r['metric']:<10} ROI {r['n_roi']:>4} 組 {r['n_clusters']:>4}  差 {r['diff']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  p {r['p']:.4f}")
    print(f"\n輸出: {path}")


if __name__ == "__main__":
    main()
