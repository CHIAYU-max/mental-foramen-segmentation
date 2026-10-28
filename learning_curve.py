import argparse
import csv
import gc
import math
import random
import statistics
import sys

import cv2
import torch

from compare_yolo_unet import GT_DIR, IMAGES_DIR, NAMES, ROOT, index_files, load_gt, pixel_metrics
from kfold_patient import OUT_DIR as PATIENT_DIR
from kfold_patient import UNET_TAG
from kfold_patient import WEIGHTS_DIR as FULL_WEIGHTS_DIR
from kfold_unet_variants import CONFIGS, build_model, fold_paths, predict_masks, train
from kfold_yolo_unet import K_FOLDS, SEED, UNET_ARGS, load_samples
from predict_unet import DEVICE

# 學習曲線:沿用 kfold_patient.py 的病人層級切分,每一折只用一部分訓練病人來訓練 U-Net,
# 觀察分數隨訓練資料量的變化,判斷再增加標註資料是否還有幫助。100% 直接沿用 kfold_patient.py 的權重。
FRACTIONS = [25, 50, 75]
OUT_DIR = ROOT / "compare_results" / "learning_curve_mf_559"
WEIGHTS_DIR = OUT_DIR / "weights"


def subset_dir(fraction):
    return OUT_DIR / f"frac{fraction}"


def write_subsets():
    # 以病人為單位抽樣;同一折內 25% ⊂ 50% ⊂ 75%,曲線才不會因為抽到不同病人而上下跳動
    with open(PATIENT_DIR / "folds.csv", newline="") as f:
        patient_of = {r["stem"]: r["patient"] for r in csv.DictReader(f)}
    for fold in range(K_FOLDS):
        train_paths = fold_paths(fold, "train", PATIENT_DIR)
        patients = sorted({patient_of[p.stem] for p in train_paths})
        random.Random(SEED + fold).shuffle(patients)
        for fraction in FRACTIONS:
            keep = set(patients[: math.ceil(len(patients) * fraction / 100)])
            yaml_dir = subset_dir(fraction) / "yaml"
            yaml_dir.mkdir(parents=True, exist_ok=True)
            subset = [p for p in train_paths if patient_of[p.stem] in keep]
            (yaml_dir / f"fold{fold}_train.txt").write_text("".join(f"{p}\n" for p in subset))
            (yaml_dir / f"fold{fold}_val.txt").write_text((PATIENT_DIR / "yaml" / f"fold{fold}_val.txt").read_text())


def weights_path(fraction, fold):
    if fraction == 100:
        return FULL_WEIGHTS_DIR / f"{UNET_TAG}_fold{fold}.pth"
    return WEIGHTS_DIR / f"{UNET_TAG}_frac{fraction}_fold{fold}.pth"


def mean_dice(model, samples, size):
    # 回傳 {類別: 逐張 Dice 的平均}
    scores = {cls: [] for cls in NAMES.values()}
    for _, image, gt in samples:
        masks = predict_masks(model, image, size)
        for cls in NAMES.values():
            scores[cls].append(pixel_metrics(masks[cls], gt[cls])["dice"])
    return {cls: statistics.mean(values) for cls, values in scores.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fractions", type=int, nargs="+", default=FRACTIONS, choices=FRACTIONS)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(K_FOLDS)))
    parser.add_argument("--same-epochs", action="store_true", help="各資料量都訓練相同的 epoch 數(預設是相同的更新步數)")
    parser.add_argument("--images", default=IMAGES_DIR, help="獨立測試集的影像資料夾")
    parser.add_argument("--gt", default=GT_DIR, help="獨立測試集的標註資料夾")
    parser.add_argument("--eval-only", action="store_true", help="不訓練,只用已訓練好的權重評分")
    args = parser.parse_args()

    # 輸出導向檔案(nohup)時預設會整塊緩衝,改成逐行寫出才能即時看到進度
    sys.stdout.reconfigure(line_buffering=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if not (PATIENT_DIR / "folds.csv").exists():
        raise SystemExit(f"找不到 {PATIENT_DIR / 'folds.csv'},請先執行 kfold_patient.py")
    write_subsets()
    config = CONFIGS[UNET_TAG]

    if not args.eval_only:
        build_model(config)
        # 先跑完一整折的所有資料量再換下一折,中途就能看到完整的一條曲線
        for fold in args.folds:
            for fraction in args.fractions:
                name = f"{UNET_TAG}_frac{fraction}"
                if (WEIGHTS_DIR / f"{name}_fold{fold}.done").exists():
                    print(f"[已完成] {name} fold={fold}")
                    continue
                # 資料變少時每個 epoch 的更新次數也變少;預設把 epoch 數等比例放大,讓總更新步數相同,
                # 這樣分數的差異才是來自資料量,而不是訓練不足
                epochs = UNET_ARGS["epochs"] if args.same_epochs else round(UNET_ARGS["epochs"] * 100 / fraction)
                train(name, config, fold, epochs, subset_dir(fraction), WEIGHTS_DIR)
                gc.collect()
                torch.cuda.empty_cache()

    test_gt = (ROOT / args.gt).resolve()
    test_samples = []
    for path in index_files((ROOT / args.images).resolve()).values():
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        test_samples.append((path, image, load_gt(path.stem, image.shape, test_gt, {})))
    val_samples = {fold: load_samples(fold_paths(fold, "val", PATIENT_DIR)) for fold in range(K_FOLDS)}

    rows = []
    for fraction in FRACTIONS + [100]:
        for fold in range(K_FOLDS):
            done = weights_path(fraction, fold).with_suffix(".done")
            if not done.exists():
                continue
            model = build_model({**config, "pretrained": False}).to(DEVICE)
            model.load_state_dict(torch.load(weights_path(fraction, fold), map_location=DEVICE)["model_state_dict"])
            model.eval()
            n_train = len(fold_paths(fold, "train", PATIENT_DIR if fraction == 100 else subset_dir(fraction)))
            row = {"fraction": fraction, "fold": fold, "n_train": n_train}
            for dataset, samples in (("cv", val_samples[fold]), ("test", test_samples)):
                for cls, dice in mean_dice(model, samples, config["size"]).items():
                    row[f"{dataset}_{cls}"] = round(dice, 4)
            rows.append(row)
            del model
    if not rows:
        raise SystemExit("還沒有任何訓練完成的權重")

    with open(OUT_DIR / "learning_curve.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    columns = [f"{dataset}_{cls}" for dataset in ("cv", "test") for cls in NAMES.values()]
    print("\n各資料量下的平均 Dice(各折的平均 ± 各折之間的標準差;cv 為該折的驗證影像,test 為獨立測試集)")
    print(f"{'資料量':<8}{'折數':>4}{'訓練張數':>8}" + "".join(f"{c:>18}" for c in columns))
    for fraction in FRACTIONS + [100]:
        items = [r for r in rows if r["fraction"] == fraction]
        if not items:
            continue
        cells = ""
        for c in columns:
            values = [r[c] for r in items]
            cells += f"{statistics.mean(values):>10.4f} ±{statistics.pstdev(values):<6.4f}"
        print(f"{str(fraction) + '%':<8}{len(items):>6}{statistics.mean(r['n_train'] for r in items):>12.0f}{cells}")
    print(f"\n逐折結果: {OUT_DIR / 'learning_curve.csv'}")
    print("[注意] 各資料量完成的折數不同時不能直接比較,請以折數相同的列為準")


if __name__ == "__main__":
    main()
