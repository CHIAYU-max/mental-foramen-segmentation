"""2026-11-14 外部資料(Tufts)上不需要標註的泛化檢查(使用者要求;伺服器 CPU,種子 0 五折,探索性)。

Tufts 沒有頦孔標註(授權未確認,只做內部實驗)。10/28 已知: 五折多數決有頦孔只有約 0.47、各折分歧很大。
這裡比較三個版本在同一批 Tufts 自動 ROI(external/tufts/rois,隨機 400 張,seed 0)上的:
  多數決有頦孔的比例、五折都有頦孔的比例、兩兩頦孔 IoU 的平均(一致程度)、先跑 2 折就停的比例。
版本: control(現行 R34/512 INT8)、median3(推論前 3×3 中值濾波)、quality-aug(r34_imagenet_aug_quality FP32)、main-fp32(對照)。
同樣的指標在我們的測試集 ROI 上也算一次當參考。注意: 這些是「模型有沒有穩定地給答案」,不是答案對不對;要知道命中率仍需 Tufts 頦孔標註。
輸出: compare_results/deploy_eval/robust/tufts_proxy.csv
"""
import csv
import itertools
import random

import cv2
import numpy as np
import torch

import deployment.deploy_pipeline as D
from deployment.cascade_infer import vote
from stage2.compare_yolo_unet import IMAGES_DIR, ROOT, index_files
from stage2.mf_postproc import apply_post
from analysis.pano_uncertainty import iou
from analysis.pano_visibility import load_test
from analysis.robust_stress import OUT


def stats(models, images, f):
    found, allf, agree, stop = [], [], [], []
    for img in images:
        preds = [m(f(img)) for m in models]
        mfs = [p["mf"] for p in preds]
        found.append(apply_post(vote(preds), "lcc")["mf"].any())
        allf.append(all(m.any() for m in mfs))
        agree.append(np.mean([iou(a, b) if a.any() and b.any() else float(a.any() == b.any()) for a, b in itertools.combinations(mfs, 2)]))
        a, b = mfs[0], mfs[1]
        stop.append(a.any() == b.any() and (not a.any() or iou(a, b) >= 0.5))
    return {"vote_has_mf": np.mean(found), "all5_have_mf": np.mean(allf), "mean_pair_iou": np.mean(agree), "stop_at_2": np.mean(stop)}


def main():
    tufts = sorted((ROOT / "external" / "tufts" / "rois").glob("*.png"))
    tufts = random.Random(0).sample(tufts, 400)
    paths = index_files((ROOT / IMAGES_DIR).resolve())
    sets = {"tufts (400 ROIs)": [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in tufts],
            "our test (137 ROIs)": [cv2.imread(str(paths[stem]), cv2.IMREAD_GRAYSCALE) for stem, gt in load_test() if gt["mf"].any()]}
    main_int8 = D.load_models(0, 8)
    main_fp32 = D.load_models(0, 8, "fp32")
    D.UNET_WEIGHTS = "r34_imagenet_aug_quality"
    quality = D.load_models(0, 8, "fp32")
    ident = lambda im: im
    variants = [("control", main_int8, ident), ("median3", main_int8, lambda im: cv2.medianBlur(im, 3)),
                ("main-fp32", main_fp32, ident), ("quality-aug", quality, ident), ("quality-aug + median3", quality, lambda im: cv2.medianBlur(im, 3))]
    rows = []
    with torch.inference_mode():
        for name, images in sets.items():
            for v, models, f in variants:
                rows.append({"data": name, "variant": v, **stats(models, images, f)})
                print(rows[-1], flush=True)
    with open(OUT / "tufts_proxy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
