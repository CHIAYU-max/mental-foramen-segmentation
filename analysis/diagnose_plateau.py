import argparse
import csv
import statistics

import cv2
import numpy as np
import torch

from stage2.compare_yolo_unet import NAMES, pixel_metrics
from stage2.kfold_unet_variants import CONFIGS, OUT_DIR, WEIGHTS_DIR, build_model, fold_paths, predict_masks
from stage2.kfold_yolo_unet import K_FOLDS, load_samples
from stage2.predict_unet import DEVICE
from stage2.rerank_candidates import BASELINE as YOLO_VARIANT

# 找出分數停滯的原因,不重新訓練:
#   A. 兩種不同架構的模型是否在同一批影像上失敗(用已有的逐張結果)
#   B. 模型在自己訓練過的影像上的分數,與驗證分數的落差
#   C. 容許邊界誤差後的分數,以及 mf 的中心位置誤差
MODEL = "r34_imagenet_aug"
FAIL_DICE = 0.5
TOLERANCES = [0, 2, 4, 8]
CENTER_LIMITS = [5, 10, 20]


def part_a(model_name):
    with open(OUT_DIR / "compare_cv.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["variant"] in (model_name, YOLO_VARIANT)]
    review = []
    print(f"\n===== A. {model_name} 與 {YOLO_VARIANT} 是否在同一批影像上失敗(交叉驗證, Dice < {FAIL_DICE} 視為失敗) =====")
    for cls in NAMES.values():
        unet = {r["filename"]: r for r in rows if r["class"] == cls and r["variant"] == model_name}
        yolo = {r["filename"]: r for r in rows if r["class"] == cls and r["variant"] == YOLO_VARIANT}
        stems = sorted(set(unet) & set(yolo))
        u = [float(unet[s]["dice"]) for s in stems]
        y = [float(yolo[s]["dice"]) for s in stems]
        both = [s for s, a, b in zip(stems, u, y) if a < FAIL_DICE and b < FAIL_DICE]
        only_unet = sum(a < FAIL_DICE <= b for a, b in zip(u, y))
        only_yolo = sum(b < FAIL_DICE <= a for a, b in zip(u, y))
        # 兩個模型若各自獨立失敗,同時失敗的張數應接近這個期望值
        expected = (len(both) + only_unet) * (len(both) + only_yolo) / len(stems)
        print(f"[{cls}] n={len(stems)}  逐張 Dice 的相關係數 {statistics.correlation(u, y):.3f}")
        print(f"  兩者都失敗 {len(both)} 張(若互相獨立預期約 {expected:.0f} 張), 只有 U-Net 失敗 {only_unet} 張, 只有 YOLO 失敗 {only_yolo} 張")
        print(f"  每張取兩者中較好的一個: 平均 Dice {statistics.mean(max(a, b) for a, b in zip(u, y)):.4f} (U-Net {statistics.mean(u):.4f}, YOLO {statistics.mean(y):.4f})")

        # 依 GT 面積分成三等分,看小目標是否特別差
        by_size = sorted(stems, key=lambda s: int(unet[s]["gt_pixels"]))
        third = len(by_size) // 3
        for label, group in (("小", by_size[:third]), ("中", by_size[third : 2 * third]), ("大", by_size[2 * third :])):
            sizes = [int(unet[s]["gt_pixels"]) for s in group]
            dice = statistics.mean(float(unet[s]["dice"]) for s in group)
            print(f"  GT 面積{label} ({sizes[0]}~{sizes[-1]} 像素): U-Net 平均 Dice {dice:.4f}")

        review += [
            {"class": cls, "filename": s, "fold": unet[s]["fold"], "unet_dice": unet[s]["dice"], "yolo_dice": yolo[s]["dice"]}
            for s in both
        ]

    review_csv = OUT_DIR / "review_candidates.csv"
    with open(review_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "filename", "fold", "unet_dice", "yolo_dice"])
        writer.writeheader()
        writer.writerows(review)
    print(f"兩個模型都失敗的影像清單(建議優先檢查標註): {review_csv}")


def tolerant_dice(pred, gt, tolerance):
    # 預測像素距離 GT 在容許範圍內就算對,反之亦然;tolerance=0 時等同一般的 Dice
    if not pred.any() or not gt.any():
        return float(pred.any() == gt.any())
    to_gt = cv2.distanceTransform((~gt).astype(np.uint8), cv2.DIST_L2, 3)
    to_pred = cv2.distanceTransform((~pred).astype(np.uint8), cv2.DIST_L2, 3)
    precision = (to_gt[pred] <= tolerance).mean()
    recall = (to_pred[gt] <= tolerance).mean()
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def part_bc(model_name):
    config = CONFIGS[model_name]
    train_dice = {cls: [] for cls in NAMES.values()}
    tolerant = {cls: {t: [] for t in TOLERANCES} for cls in NAMES.values()}
    center_errors, radii, no_center = [], [], 0

    for fold in range(K_FOLDS):
        model = build_model({**config, "pretrained": False}).to(DEVICE)
        model.load_state_dict(torch.load(WEIGHTS_DIR / f"{model_name}_fold{fold}.pth", map_location=DEVICE)["model_state_dict"])
        model.eval()

        for _, image, gt in load_samples(fold_paths(fold, "train")):
            masks = predict_masks(model, image, config["size"])
            for cls in NAMES.values():
                train_dice[cls].append(pixel_metrics(masks[cls], gt[cls])["dice"])

        for _, image, gt in load_samples(fold_paths(fold, "val")):
            masks = predict_masks(model, image, config["size"])
            for cls in NAMES.values():
                for t in TOLERANCES:
                    tolerant[cls][t].append(tolerant_dice(masks[cls], gt[cls], t))
            if gt["mf"].any():
                radii.append((gt["mf"].sum() / np.pi) ** 0.5)
                if masks["mf"].any():
                    center_errors.append(float(np.linalg.norm(np.argwhere(masks["mf"]).mean(0) - np.argwhere(gt["mf"]).mean(0))))
                else:
                    no_center += 1
        print(f"fold {fold} 完成")
        del model

    print(f"\n===== B. {model_name} 在訓練過的影像與沒看過的影像上的 Dice =====")
    for cls in NAMES.values():
        val = statistics.mean(tolerant[cls][0])
        print(f"[{cls}] 訓練影像 {statistics.mean(train_dice[cls]):.4f}   驗證影像 {val:.4f}   落差 {statistics.mean(train_dice[cls]) - val:+.4f}")

    print(f"\n===== C. 容許邊界誤差後的 Dice(驗證影像;容許 0 像素即一般的 Dice) =====")
    print(f"{'class':<6}" + "".join(f"{f'容許 {t} px':>14}" for t in TOLERANCES))
    for cls in NAMES.values():
        print(f"{cls:<6}" + "".join(f"{statistics.mean(tolerant[cls][t]):>16.4f}" for t in TOLERANCES))

    n = len(radii)
    print(f"\nmf 中心位置誤差(驗證影像 {n} 張;GT 等效半徑中位數 {statistics.median(radii):.1f} 像素)")
    print(f"  沒有預測: {no_center} 張 ({no_center / n:.1%})")
    print(f"  有預測者的誤差中位數: {statistics.median(center_errors):.1f} 像素")
    for limit in CENTER_LIMITS:
        hit = sum(e <= limit for e in center_errors)
        print(f"  誤差在 {limit} 像素以內: {hit} 張 ({hit / n:.1%})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL, choices=list(CONFIGS))
    parser.add_argument("--skip-predict", action="store_true", help="只做 A(不需要重新預測)")
    args = parser.parse_args()

    part_a(args.model)
    if not args.skip_predict:
        part_bc(args.model)


if __name__ == "__main__":
    main()
