import argparse
import csv
import gc
import sys

import cv2
import torch

from compare_yolo_unet import GT_DIR, IMAGES_DIR, NAMES, ROOT, index_files, load_gt, pixel_metrics
from eval_kfold_variants import FIELDS, summarize
from kfold_patient import OUT_DIR as PATIENT_DIR
from kfold_patient import UNET_TAG, YOLO_TAG, save_masks
from kfold_unet_variants import CONFIGS, build_model, fold_paths, predict_masks, train
from kfold_yolo_unet import K_FOLDS, load_samples
from predict_unet import DEVICE

# 消融實驗:在 kfold_patient.py 的病人層級切分上,把最終模型(r34_imagenet_aug)的設計逐一拿掉,
# 量化每個設計的貢獻。訓練設定與 kfold_patient.py 完全相同,只差下表中的項目。
#   unet_scratch:     原本的 U-Net 結構、從零訓練、基本增強(最初的版本)
#   scratch_aug:      原本的 U-Net 結構、從零訓練、強增強
#   r34_scratch_aug:  ResNet34 encoder、從零訓練、強增強(與最終模型只差在有無預訓練)
#   r34_imagenet:     ResNet34 encoder、ImageNet 預訓練、基本增強(與最終模型只差在增強)
ABLATIONS = {
    "unet_scratch": dict(arch="unet", pretrained=False, aug="basic", size=(512, 512), lr=1e-3),
    "scratch_aug": CONFIGS["scratch_aug"],
    "r34_scratch_aug": dict(arch="resnet34", pretrained=False, aug="strong", size=(512, 512), lr=1e-3),
    "r34_imagenet": CONFIGS["r34_imagenet"],
}

OUT_DIR = ROOT / "compare_results" / "ablation_patient_mf_559"
WEIGHTS_DIR = OUT_DIR / "weights"


def evaluate(name, config, folds, dataset, test_samples):
    rows = []
    for fold in folds:
        model = build_model({**config, "pretrained": False}).to(DEVICE)
        model.load_state_dict(torch.load(WEIGHTS_DIR / f"{name}_fold{fold}.pth", map_location=DEVICE)["model_state_dict"])
        model.eval()
        samples = load_samples(fold_paths(fold, "val", PATIENT_DIR)) if dataset == "cv" else test_samples
        for path, image, gt in samples:
            masks = predict_masks(model, image, config["size"])
            # 與 kfold_patient.py 相同的格式存下遮罩,供 paper_metrics.py 計算其他指標
            save_masks(masks, OUT_DIR / "masks" / dataset / name / f"fold{fold}" / f"{path.stem}.png")
            for cls in NAMES.values():
                row = {"dataset": dataset, "fold": fold, "filename": path.stem, "variant": name, "class": cls}
                rows.append({**row, **pixel_metrics(masks[cls], gt[cls])})
        del model
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablations", nargs="+", default=list(ABLATIONS), choices=list(ABLATIONS))
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(K_FOLDS)))
    parser.add_argument("--images", default=IMAGES_DIR, help="獨立測試集的影像資料夾")
    parser.add_argument("--gt", default=GT_DIR, help="獨立測試集的標註資料夾")
    parser.add_argument("--eval-only", action="store_true", help="不訓練,只用已訓練好的折評分")
    args = parser.parse_args()

    # 輸出導向檔案(nohup)時預設會整塊緩衝,改成逐行寫出才能即時看到進度
    sys.stdout.reconfigure(line_buffering=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for dataset in ("cv", "test"):
        if not (PATIENT_DIR / f"compare_{dataset}.csv").exists():
            raise SystemExit(f"找不到 {PATIENT_DIR / f'compare_{dataset}.csv'},請先跑完 kfold_patient.py")

    if not args.eval_only:
        # 預訓練權重下載失敗或設定有誤時,在開始訓練前就會發現
        for name in args.ablations:
            build_model(ABLATIONS[name])
        # 先跑完一整折的所有版本再換下一折,中途的結果也能互相比較
        for fold in args.folds:
            for name in args.ablations:
                if (WEIGHTS_DIR / f"{name}_fold{fold}.done").exists():
                    print(f"[已完成] {name} fold={fold}")
                    continue
                train(name, ABLATIONS[name], fold, 100, PATIENT_DIR, WEIGHTS_DIR)
                gc.collect()
                torch.cuda.empty_cache()

    test_gt = (ROOT / args.gt).resolve()
    test_samples = []
    for path in index_files((ROOT / args.images).resolve()).values():
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        test_samples.append((path, image, load_gt(path.stem, image.shape, test_gt, {})))

    for dataset in ("cv", "test"):
        # 最終模型與 YOLO 的結果直接沿用 kfold_patient.py
        with open(PATIENT_DIR / f"compare_{dataset}.csv", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["variant"] in (UNET_TAG, YOLO_TAG)]
        for name, config in ABLATIONS.items():
            folds = [f for f in range(K_FOLDS) if (WEIGHTS_DIR / f"{name}_fold{f}.done").exists()]
            if folds:
                rows += evaluate(name, config, folds, dataset, test_samples)
                print(f"[{dataset}] {name}: 已完成的折 {folds}")
        merged_csv = OUT_DIR / f"compare_{dataset}.csv"
        with open(merged_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        summarize(dataset, merged_csv, UNET_TAG)
    print("\ndiff 為各版本減去最終模型(r34_imagenet_aug),負值即拿掉該設計後下降的幅度")
    print("[注意] 尚未跑完五折的版本只包含已完成的折,請以五折都完成後的結果為準")


if __name__ == "__main__":
    main()
