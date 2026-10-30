import argparse
import gc
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from stage2.compare_yolo_unet import GT_DIR, IMAGES_DIR, IOU, ROOT, index_files, load_gt
from stage2.eval_kfold_variants import MIN_CONF
from stage2.kfold_patient import OUT_DIR as PATIENT_DIR
from stage2.kfold_patient import UNET_TAG, YOLO_TAG, save_masks
from stage2.kfold_patient import yolo_dir as seed0_yolo_dir
from stage2.kfold_unet_variants import CONFIGS, build_model, fold_paths, predict_masks, train
from stage2.kfold_yolo_unet import K_FOLDS, SEED, UNET_ARGS, YOLO_ARGS, YOLO_MODEL, load_samples
from analysis.paper_metrics import collect
from stage2.predict_unet import DEVICE
from stage2.rerank_candidates import get_candidates

# 以不同的隨機種子重複訓練最終的 U-Net 與 YOLO(切分與設定完全相同),
# 量化訓練本身的隨機波動,確認 U-Net 與 YOLO 的比較結果不是單一次訓練的偶然。
# 種子 0 直接沿用 kfold_patient.py 的結果。
SEEDS = [1, 2]
PROJECT = "seeds_patient_mf_559"
OUT_DIR = ROOT / "compare_results" / PROJECT
WEIGHTS_DIR = OUT_DIR / "weights"


def yolo_dir(seed, fold):
    return seed0_yolo_dir(fold) if seed == SEED else ROOT / "runs" / "segment" / PROJECT / f"yolo_seed{seed}_fold{fold}"


def unet_name(seed):
    return f"{UNET_TAG}_seed{seed}"


def source(seed, tag):
    # (遮罩資料夾, 版本名稱)
    if seed == SEED:
        return PATIENT_DIR, tag
    return OUT_DIR, f"{tag}_seed{seed}"


def predict(seed, fold, val, test):
    for dataset, samples in (("cv", val), ("test", test)):
        listing = OUT_DIR / f"eval_{dataset}_seed{seed}_fold{fold}.txt"
        listing.write_text("".join(f"{path}\n" for path, _, _ in samples))
        yolo = YOLO(yolo_dir(seed, fold) / "weights" / "best.pt")
        for result in yolo.predict(source=str(listing), conf=MIN_CONF, iou=IOU, max_det=50, retina_masks=True, stream=True, verbose=False):
            shape = tuple(result.orig_shape)
            masks = {cls: items[0][1] if items else np.zeros(shape, dtype=bool) for cls, items in get_candidates(result, shape).items()}
            save_masks(masks, OUT_DIR / "masks" / dataset / f"{YOLO_TAG}_seed{seed}" / f"fold{fold}" / f"{Path(result.path).stem}.png")
        del yolo

        unet = build_model({**CONFIGS[UNET_TAG], "pretrained": False}).to(DEVICE)
        unet.load_state_dict(torch.load(WEIGHTS_DIR / f"{unet_name(seed)}_fold{fold}.pth", map_location=DEVICE)["model_state_dict"])
        unet.eval()
        for path, image, _ in samples:
            masks = predict_masks(unet, image, CONFIGS[UNET_TAG]["size"])
            save_masks(masks, OUT_DIR / "masks" / dataset / unet_name(seed) / f"fold{fold}" / f"{path.stem}.png")
        del unet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--eval-only", action="store_true", help="不訓練,只用已訓練好的模型評分")
    args = parser.parse_args()

    # 輸出導向檔案(nohup)時預設會整塊緩衝,改成逐行寫出才能即時看到進度
    sys.stdout.reconfigure(line_buffering=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.eval_only:
        build_model(CONFIGS[UNET_TAG])
        for seed in args.seeds:
            for fold in range(K_FOLDS):
                # 訓練結束才會產生 results.png / .done;已訓練完的直接沿用
                if not (yolo_dir(seed, fold) / "results.png").exists():
                    model = YOLO(YOLO_MODEL)
                    data_yaml = PATIENT_DIR / "yaml" / f"fold{fold}.yaml"
                    model.train(data=str(data_yaml), project=PROJECT, name=f"yolo_seed{seed}_fold{fold}", seed=seed, save=True, plots=True, exist_ok=True, **YOLO_ARGS)
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                if not (WEIGHTS_DIR / f"{unet_name(seed)}_fold{fold}.done").exists():
                    train(unet_name(seed), CONFIGS[UNET_TAG], fold, UNET_ARGS["epochs"], PATIENT_DIR, WEIGHTS_DIR, seed=seed)
                    gc.collect()
                    torch.cuda.empty_cache()

    test_gt = (ROOT / GT_DIR).resolve()
    test = []
    for stem, path in index_files((ROOT / IMAGES_DIR).resolve()).items():
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        test.append((path, image, load_gt(stem, image.shape, test_gt, {})))

    done = [
        s for s in args.seeds
        if all((yolo_dir(s, f) / "results.png").exists() and (WEIGHTS_DIR / f"{unet_name(s)}_fold{f}.done").exists() for f in range(K_FOLDS))
    ]
    cv_samples = []
    for fold in range(K_FOLDS):
        val = load_samples(fold_paths(fold, "val", PATIENT_DIR))
        for seed in done:
            predict(seed, fold, val, test)
        cv_samples += [(fold, path.stem, gt) for path, _, gt in val]
    test_samples = [(path.stem, gt) for path, _, gt in test]

    # 每個種子的五折結果;測試集為五折多數決
    seeds = [SEED] + done
    metrics = {}
    for seed in seeds:
        for name, tag in (("U-Net", UNET_TAG), ("YOLO", YOLO_TAG)):
            root, variant = source(seed, tag)
            for dataset, rows in (("cv", collect(root, variant, "cv", cv_samples, [], False)), ("test", collect(root, variant, "test", [], test_samples, True))):
                hits = [r["mf_hit_1R"] for r in rows if "mf_hit_1R" in r]
                metrics[(seed, name, dataset)] = {
                    "mf Dice": np.mean([r["mf_dice"] for r in rows]),
                    "mf 命中1R": np.mean(hits),
                    "ian Dice": np.mean([r["ian_dice"] for r in rows]),
                }

    keys = ["mf Dice", "mf 命中1R", "ian Dice"]
    print(f"\n已完成的種子: {seeds}(種子 {SEED} 為 kfold_patient.py 的結果)")
    for dataset, title in (("cv", "交叉驗證"), ("test", "測試集(五折多數決)")):
        print(f"\n===== {title} =====")
        print(f"{'模型':<8}" + "".join(f"{f'seed {s}':>30}" for s in seeds) + f"{'平均 ± 標準差':>36}")
        for name in ("U-Net", "YOLO", "U-Net - YOLO"):
            for k in keys:
                if name == "U-Net - YOLO":
                    values = [metrics[(s, "U-Net", dataset)][k] - metrics[(s, "YOLO", dataset)][k] for s in seeds]
                else:
                    values = [metrics[(s, name, dataset)][k] for s in seeds]
                sd = statistics.stdev(values) if len(values) > 1 else 0.0
                fmt = "+.4f" if name == "U-Net - YOLO" else ".4f"
                cells = "".join(f"{format(v, fmt):>24}" for v in values)
                print(f"{name:<14}{k:<10}{cells}{format(statistics.mean(values), fmt):>22} ± {sd:.4f}")
    print("\n「U-Net - YOLO」在每個種子都同號,且平均遠大於標準差,比較結果才算穩定")


if __name__ == "__main__":
    main()
