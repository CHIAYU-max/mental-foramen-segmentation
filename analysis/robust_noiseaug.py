"""2026-11-18 雜訊增強權重的壓力測試(使用者要求;伺服器 CPU,種子 0)。

acc_noise(r34_imagenet_aug_noise,只加雜訊增強,種子 0)的 CV 命中對主結果 −0.7 點,符合 run_acc_compute.sh 事先定的判準
(沒有掉超過約 1 點 → 再看雜訊下的命中)。條件與 robust_remedy.py 相同(同一組合成變化、同一個亂數種子),比較:
  main-int8          現行 R34/512 INT8 串接式聯集(= robust_remedy 的 control)
  main-int8+median3  現行模型 + 推論前 3×3 中值濾波(不用重訓的補救,--pre median3)
  noise-int8         雜訊增強權重,同樣的 nncf INT8 量化(各折訓練集 300 張校正)
  main-fp32 / noise-fp32   同一組權重的 FP32(排除 INT8 的影響)
判讀(事先定): 雜訊增強要在 noise sd 5 / sd 10 比 main-int8+median3 好,而且原圖不比 main-int8 差超過約 1 點,才值得補種子。
另以全景片為組 bootstrap(同 robust_stress.py) 算 noise-int8 − main-int8 與 noise-int8 − main-int8+median3 的命中差。
輸出: compare_results/deploy_eval/robust/noiseaug.csv、noiseaug_per_roi.csv、noiseaug_diff.csv
"""
import csv

import cv2
import numpy as np
import torch

import deployment.deploy_pipeline as D
from stage2.compare_yolo_unet import IMAGES_DIR, ROOT, index_files
from analysis.paper_metrics import image_metrics
from analysis.pano_visibility import load_test
from analysis.robust_stress import CONDITIONS, OUT
from analysis.robust_remedy import CONDS, FILTERS
from analysis.e2e_seeds import SIDE
from analysis.seed_bootstrap import cluster_test

NOISE = "r34_imagenet_aug_noise"


def diff_test(a, b, stems, rng):
    # 與 robust_stress.py 相同: 以全景片(stem 去掉左右側)為組,seed_bootstrap.cluster_test
    clusters = {}
    for stem, x, y in zip(stems, a, b):
        c = SIDE.sub("", stem)
        s0, n0 = clusters.get(c, (0.0, 0))
        clusters[c] = (s0 + x - y, n0 + 1)
    return cluster_test(list(clusters.values()), rng)

def main():
    D.CASCADE_OUT = "union"
    paths = index_files((ROOT / IMAGES_DIR).resolve())
    test = [(stem, gt, cv2.imread(str(paths[stem]), cv2.IMREAD_GRAYSCALE)) for stem, gt in load_test() if gt["mf"].any()]
    stems = [stem for stem, _, _ in test]
    print(f"測試 ROI {len(test)},全景片 {len({SIDE.sub('', s) for s in stems})}", flush=True)
    main_int8 = D.load_models(0, 8)
    main_fp32 = D.load_models(0, 8, "fp32")
    noise_int8 = D.load_models(0, 8, "int8", tag=NOISE)
    noise_fp32 = D.load_models(0, 8, "fp32", tag=NOISE)
    variants = [("main-int8", main_int8, FILTERS["control"]), ("main-int8+median3", main_int8, FILTERS["median3"]),
                ("noise-int8", noise_int8, FILTERS["control"]), ("main-fp32", main_fp32, FILTERS["control"]),
                ("noise-fp32", noise_fp32, FILTERS["control"])]
    rows, per_roi = [], []
    hits = {}
    with torch.inference_mode():
        for cond in CONDS:
            rng = np.random.default_rng(1)
            xs = [CONDITIONS[cond](img, rng) for _, _, img in test]
            for name, models, f in variants:
                res = []
                for (stem, gt, _), x in zip(test, xs):
                    pred, n = D.cascade(models, f(x), D.Clock())
                    m = image_metrics(pred, gt)
                    res.append((m["mf_hit_1R"], m["mf_dice"], n))
                    per_roi.append({"condition": cond, "variant": name, "stem": stem, "hit": m["mf_hit_1R"], "mf_dice": m["mf_dice"], "models_run": n})
                r = np.array(res, float)
                hits[cond, name] = r[:, 0]
                rows.append({"condition": cond, "variant": name, "hit": r[:, 0].mean(), "mf_dice": np.nanmean(r[:, 1]), "models_run": r[:, 2].mean()})
                print(rows[-1], flush=True)
    diffs = []
    rng = np.random.default_rng(0)
    for cond in CONDS:
        for a, b in [("noise-int8", "main-int8"), ("noise-int8", "main-int8+median3"), ("noise-fp32", "main-fp32")]:
            d, lo, hi, p = diff_test(hits[cond, a], hits[cond, b], stems, rng)
            diffs.append({"condition": cond, "a": a, "b": b, "hit_diff": d, "ci_lo": lo, "ci_hi": hi, "p": p})
            print(f"{cond:14s} {a} − {b}: {d * 100:+.2f} 點 [{lo * 100:+.2f}, {hi * 100:+.2f}] p {p:.4f}", flush=True)
    for name, data in [("noiseaug.csv", rows), ("noiseaug_per_roi.csv", per_roi), ("noiseaug_diff.csv", diffs)]:
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)


if __name__ == "__main__":
    main()
