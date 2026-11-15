"""2026-11-14 壓力測試的補救(使用者要求;伺服器 CPU,種子 0,探索性)。

robust_stress.py 顯示部署的第二階段對雜訊最敏感(sd 5 掉約 9 點、sd 10 掉約 37 點),其次是模糊。這裡在同樣的合成變化下比較:
  control        現行 R34/512 INT8 串接式聯集
  median3 / gauss0.8 / bilateral / nlmeans   推論前先去雜訊(不用重訓),再給現行模型
  quality-aug    11/5 訓練的 r34_imagenet_aug_quality(訓練時加 gamma / 對比 / 模糊 / 雜訊增強;原分布測試命中少 2.9 點,所以沒採用),FP32 串接式聯集
  main-fp32      現行權重的 FP32(quality-aug 的對照,排除 INT8 的影響)
補救在原分布上不能變差才有用;所有變化與濾波參數事先固定。
輸出: compare_results/deploy_eval/robust/remedy.csv
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

CONDS = ["original", "noise sd 5", "noise sd 10", "blur sd 1", "blur sd 2", "jpeg q30", "contrast x0.5"]
FILTERS = {
    "control": lambda im: im,
    "median3": lambda im: cv2.medianBlur(im, 3),
    "gauss0.8": lambda im: cv2.GaussianBlur(im, (0, 0), 0.8),
    "bilateral": lambda im: cv2.bilateralFilter(im, 5, 25, 3),
    "nlmeans": lambda im: cv2.fastNlMeansDenoising(im, None, 7, 7, 21),
}


def main():
    D.CASCADE_OUT = "union"
    paths = index_files((ROOT / IMAGES_DIR).resolve())
    test = [(stem, gt, cv2.imread(str(paths[stem]), cv2.IMREAD_GRAYSCALE)) for stem, gt in load_test() if gt["mf"].any()]
    main_int8 = D.load_models(0, 8)
    main_fp32 = D.load_models(0, 8, "fp32")
    D.UNET_WEIGHTS = "r34_imagenet_aug_quality"
    quality = D.load_models(0, 8, "fp32")
    variants = [(name, main_int8, f) for name, f in FILTERS.items()] + [("main-fp32", main_fp32, FILTERS["control"]), ("quality-aug", quality, FILTERS["control"])]
    rows = []
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
                r = np.array(res, float)
                rows.append({"condition": cond, "variant": name, "hit": r[:, 0].mean(), "mf_dice": np.nanmean(r[:, 1]), "models_run": r[:, 2].mean()})
                print(rows[-1], flush=True)
    with open(OUT / "remedy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
