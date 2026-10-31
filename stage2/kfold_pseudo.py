import argparse
import csv
import gc
import sys
from pathlib import Path

import cv2
import torch

from stage2.compare_yolo_unet import ROOT
from stage2.kfold_pano import PROJECT, SUMMARY_KEYS, evaluate
from stage2.kfold_patient import save_masks
from stage2.kfold_unet_variants import CONFIGS, build_model, fold_paths, predict_masks, train
from stage2.kfold_yolo_unet import K_FOLDS, SEED, UNET_ARGS
from analysis.pano_visibility import load_test
from stage2.predict_unet import DEVICE
from analysis.unlabeled_shift import OUT

# 半監督訓練(2026-10-28,GPU): r34_imagenet_aug 的設定與超參數完全不變,只在每一折的訓練資料加上
#   pseudo_label.py 產生的假標註 ROI(第 k 折只加第 k 折老師的假標註)。驗證集、測試集、評分都與 kfold_pano 相同。
#   輸出名稱 r34_imagenet_aug_pl(種子不是 0 時加 _s<種子>),權重與遮罩放在 kfold_pano_mf_559 底下,彙總 summary_<名稱>.csv。
CONFIG = "r34_imagenet_aug"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(K_FOLDS)))
    parser.add_argument("--epochs", type=int, default=UNET_ARGS["epochs"], help="試跑用")
    parser.add_argument("--name", default=f"{CONFIG}_pl", help="試跑時換名稱")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    out_dir = ROOT / "compare_results" / PROJECT
    tag = args.name + (f"_s{args.seed}" if args.seed != SEED else "")
    config = CONFIGS[CONFIG]
    test = load_test()
    test_paths = [Path(line) for line in (out_dir / "yaml" / "test.txt").read_text().splitlines() if line]

    for fold in args.folds:
        done = out_dir / "weights" / f"{tag}_fold{fold}.done"
        extra = [Path(line) for line in (OUT / f"pseudo_fold{fold}.txt").read_text().splitlines() if line]
        if done.exists():
            print(f"[已完成] {tag} fold={fold}")
        else:
            print(f"[開始訓練] {tag} fold={fold}: 真標註 {len(fold_paths(fold, 'train', out_dir))} + 假標註 {len(extra)} 張")
            train(tag, config, fold, args.epochs, out_dir, out_dir / "weights", seed=args.seed, extra=extra)
            gc.collect()
            torch.cuda.empty_cache()
        unet = build_model({**config, "pretrained": False}).to(DEVICE)
        unet.load_state_dict(torch.load(out_dir / "weights" / f"{tag}_fold{fold}.pth", map_location=DEVICE)["model_state_dict"])
        unet.eval()
        for dataset, paths in (("cv", fold_paths(fold, "val", out_dir)), ("test", test_paths)):
            for path in paths:
                image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                save_masks(predict_masks(unet, image, config["size"]), out_dir / "masks" / dataset / tag / f"fold{fold}" / f"{path.stem}.png")
        del unet
        torch.cuda.empty_cache()

    summary = evaluate(tag, args.folds, out_dir, test)
    for r in summary:
        print(f"{r['model']:<28}{r['setting']:<16}{r['n']:>5}" + "".join(f"{r[k]:>14.4f}" for k in SUMMARY_KEYS))
    path = out_dir / f"summary_{tag}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"彙總結果: {path}")


if __name__ == "__main__":
    main()
