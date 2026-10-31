import argparse
import re
import sys

import cv2
import torch

from stage2.compare_yolo_unet import ROOT
from stage2.kfold_pano import PROJECT
from stage2.kfold_patient import save_masks
from stage2.kfold_unet_variants import CONFIGS, build_model, predict_masks
from stage2.kfold_yolo_unet import K_FOLDS
from stage2.predict_unet import DEVICE
from analysis.unlabeled_pool import OUT_IMAGES

# 沒有標註的 ROI 池(unlabeled_pool.py)的推論(2026-10-28,GPU): kfold_pano 的 U-Net 每個種子、每一折都推論一次,
#   遮罩存到 compare_results/kfold_pano_mf_559/masks/unlabeled/<設定>/fold<k>/(位元旗標,與 cv / test 相同),
#   給 unlabeled_shift.py(分布檢查)與 pseudo_label.py(假標註)使用。推論與 kfold_pano.predict_fold 相同(沒有後處理)。
TAGS = ["r34_imagenet_aug"] + [f"r34_imagenet_aug_s{s}" for s in range(1, 5)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", default=TAGS)
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    out_dir = ROOT / "compare_results" / PROJECT
    paths = sorted(OUT_IMAGES.glob("*.jpg"))
    print(f"沒有標註的 ROI {len(paths)} 張")
    for tag in args.tags:
        config = CONFIGS[re.sub(r"_s\d+$", "", tag)]
        for fold in range(K_FOLDS):
            unet = build_model({**config, "pretrained": False}).to(DEVICE)
            unet.load_state_dict(torch.load(out_dir / "weights" / f"{tag}_fold{fold}.pth", map_location=DEVICE)["model_state_dict"])
            unet.eval()
            for path in paths:
                image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                save_masks(predict_masks(unet, image, config["size"]), out_dir / "masks" / "unlabeled" / tag / f"fold{fold}" / f"{path.stem}.png")
            del unet
            torch.cuda.empty_cache()
        print(f"[完成] {tag}")


if __name__ == "__main__":
    main()
