import argparse
import csv

import cv2
import torch

from stage2.compare_yolo_unet import NAMES
from analysis.diagnose_plateau import MODEL
from stage2.kfold_unet_variants import CONFIGS, OUT_DIR, WEIGHTS_DIR, build_model, fold_paths, predict_masks
from stage2.kfold_yolo_unet import K_FOLDS, load_samples
from stage2.predict_unet import DEVICE
from data.show_failures import draw

# 把 diagnose_plateau.py 找出的「兩個模型都失敗」的影像畫成 GT 與 U-Net 預測的疊圖,供人工檢查標註
REVIEW_CSV = OUT_DIR / "review_candidates.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL, choices=list(CONFIGS))
    args = parser.parse_args()

    if not REVIEW_CSV.exists():
        raise SystemExit(f"找不到 {REVIEW_CSV},請先執行 diagnose_plateau.py")
    with open(REVIEW_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    # {檔名: {類別: 該列}}
    selected = {}
    for r in rows:
        selected.setdefault(r["filename"], {})[r["class"]] = r

    out_dir = OUT_DIR / "review"
    for cls in NAMES.values():
        (out_dir / cls).mkdir(parents=True, exist_ok=True)

    config = CONFIGS[args.model]
    count = 0
    for fold in range(K_FOLDS):
        paths = [p for p in fold_paths(fold, "val") if p.stem in selected]
        if not paths:
            continue
        model = build_model({**config, "pretrained": False}).to(DEVICE)
        model.load_state_dict(torch.load(WEIGHTS_DIR / f"{args.model}_fold{fold}.pth", map_location=DEVICE)["model_state_dict"])
        model.eval()
        for path, image, gt in load_samples(paths):
            pred = predict_masks(model, image, config["size"])
            scores = " | ".join(
                f"{cls} unet {float(r['unet_dice']):.2f} yolo {float(r['yolo_dice']):.2f}" for cls, r in selected[path.stem].items()
            )
            canvas = draw(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), gt, pred, f"{path.stem} fold{fold} | both fail: {scores}")
            # 檔名以 U-Net 的 Dice 開頭,依檔名排序即由差到好
            for cls, r in selected[path.stem].items():
                cv2.imwrite(str(out_dir / cls / f"dice{float(r['unet_dice']):.2f}_{path.stem}.jpg"), canvas)
            count += 1
        del model
    print(f"已輸出 {count} 張影像的疊圖到: {out_dir}")
    print("  mf/ 與 ian/ 分別是該類別兩個模型都失敗的影像;標題列只列出失敗的類別")


if __name__ == "__main__":
    main()
