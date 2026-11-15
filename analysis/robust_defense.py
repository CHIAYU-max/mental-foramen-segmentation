"""2026-11-14 不用重訓的防禦性前處理(使用者要求;伺服器 CPU,5 種子,探索性)。

robust_stress.py / robust_remedy.py(種子 0)顯示: 3×3 中值濾波能擋雜訊、zscore 能擋對比壓縮。這裡用 5 種子確認
「median3」與「median3 + zscore」在原分布上不變差,以及在雜訊 / 對比 / 模糊下的效果。現行 R34/512 INT8 串接式聯集 + lcc。
差 = 該條件該前處理 − 原圖不加前處理,以病人為組的 bootstrap。
輸出: compare_results/deploy_eval/robust/defense.csv
"""
import csv
from collections import defaultdict

import cv2
import numpy as np
import torch

import deployment.deploy_pipeline as D
from stage2.compare_yolo_unet import IMAGES_DIR, ROOT, index_files
from analysis.e2e_seeds import SIDE
from analysis.paper_metrics import image_metrics
from analysis.pano_visibility import load_test
from stage2.quality_methods import train_stats, zscore
from analysis.robust_stress import CONDITIONS, OUT
from analysis.seed_bootstrap import cluster_test

CONDS = ["original", "noise sd 5", "noise sd 10", "contrast x0.5", "blur sd 2", "jpeg q30"]


class Pre:
    def __init__(self, model, fn):
        self.model, self.fn = model, fn

    def __call__(self, gray):
        return self.model(self.fn(gray))


def main():
    D.CASCADE_OUT = "union"
    paths = index_files((ROOT / IMAGES_DIR).resolve())
    test = [(stem, gt, cv2.imread(str(paths[stem]), cv2.IMREAD_GRAYSCALE)) for stem, gt in load_test() if gt["mf"].any()]
    stats = [train_stats(k)[1:] for k in range(5)]
    med = lambda im: cv2.medianBlur(im, 3)
    per = defaultdict(lambda: defaultdict(list))
    with torch.inference_mode():
        for seed in range(5):
            models = D.load_models(seed, 8)
            sets = {"none": models, "median3": [Pre(m, med) for m in models],
                    "median3+zscore": [Pre(m, lambda im, k=k: zscore(med(im), *stats[k])) for k, m in enumerate(models)]}
            for name in CONDS:
                rng = np.random.default_rng(1000 * seed + 1)
                for stem, gt, img in test:
                    x = CONDITIONS[name](img, rng)
                    for remedy, ms in sets.items():
                        pred, n = D.cascade(ms, x, D.Clock())
                        m = image_metrics(pred, gt)
                        per[(name, remedy)][stem].append((m["mf_hit_1R"], m["mf_dice"], n))
                print(f"seed {seed} {name}", flush=True)
    mean = {k: {stem: np.nanmean(np.array(v, float), axis=0) for stem, v in d.items()} for k, d in per.items()}
    ref = mean[("original", "none")]
    rng = np.random.default_rng(0)
    rows = []
    for (name, remedy), d in mean.items():
        row = {"condition": name, "remedy": remedy, "hit": np.mean([v[0] for v in d.values()]), "mf_dice": np.nanmean([v[1] for v in d.values()]),
               "models_run": np.mean([v[2] for v in d.values()])}
        for i, key in ((0, "hit"), (1, "mf_dice")):
            clusters = {}
            for stem, v in d.items():
                diff = v[i] - ref[stem][i]
                if diff == diff:
                    c = SIDE.sub("", stem)
                    s0, n0 = clusters.get(c, (0.0, 0))
                    clusters[c] = (s0 + diff, n0 + 1)
            est, lo, hi, p = cluster_test(list(clusters.values()), rng)
            row.update({f"{key}_diff": est, f"{key}_lo": lo, f"{key}_hi": hi, f"{key}_p": p})
        rows.append(row)
        print(f"{name:<16}{remedy:<16} hit {row['hit']:.4f} ({100 * row['hit_diff']:+.2f} [{100 * row['hit_lo']:+.2f}, {100 * row['hit_hi']:+.2f}])"
              f" dice {row['mf_dice']:.4f} ({row['mf_dice_diff']:+.4f} [{row['mf_dice_lo']:+.4f}, {row['mf_dice_hi']:+.4f}]) models {row['models_run']:.2f}", flush=True)
    with open(OUT / "defense.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
