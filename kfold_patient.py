import argparse
import csv
import gc
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

from compare_preprocessing import collect_images
from compare_yolo_unet import GT_DIR, IMAGES_DIR, IOU, NAMES, ROOT, index_files, load_gt, pixel_metrics
from eval_kfold_variants import FIELDS, MIN_CONF, summarize
from kfold_unet_variants import CONFIGS, build_model, fold_paths, predict_masks, train
from kfold_yolo_unet import K_FOLDS, SEED, SPLIT_SEED, UNET_ARGS, YOLO_ARGS, YOLO_MODEL, load_samples
from predict_unet import DEVICE
from rerank_candidates import get_candidates

# 以「病人」為單位的 K 折:同一位病人的左右側一定在同一折,避免訓練與驗證看到同一個人。
# 病人編號來自 recover_patient_ids.py 產生的 patient_map.csv。
# 比較先前選出的兩個模型: YOLO(各類別取信心最高的候選)與 ResNet34 預訓練 + 強增強的 U-Net。
DATASET = "datasets/20260909/mf_559__unsharp_mask"
PATIENT_MAP = ROOT / "patient_map.csv"
PROJECT = "kfold_patient_mf_559"
YOLO_TAG = "yolo_top1"
UNET_TAG = "r34_imagenet_aug"

OUT_DIR = ROOT / "compare_results" / PROJECT
WEIGHTS_DIR = OUT_DIR / "weights"


def build_folds(images):
    with open(PATIENT_MAP, newline="") as f:
        info = {r["stem"]: r for r in csv.DictReader(f)}
    missing = sorted(set(images) - set(info))
    if missing:
        raise SystemExit(f"{len(missing)} 張影像不在 {PATIENT_MAP} 中(前5: {missing[:5]}),請重新執行 recover_patient_ids.py")

    patients = {}
    for stem in sorted(images):
        patients.setdefault((info[stem]["batch"], info[stem]["patient"]), []).append(stem)

    # 兩批資料各自分配,每一折的批次比例才會接近;每位病人放進目前影像數最少的那一折
    rng = random.Random(SPLIT_SEED)
    folds = [[] for _ in range(K_FOLDS)]
    for batch in sorted({b for b, _ in patients}):
        keys = sorted(k for k in patients if k[0] == batch)
        rng.shuffle(keys)
        for key in sorted(keys, key=lambda k: -len(patients[k])):
            min(folds, key=len).extend(patients[key])

    with open(OUT_DIR / "folds.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stem", "patient", "fold"])
        writer.writerows(sorted((s, info[s]["patient"], i) for i, fold in enumerate(folds) for s in fold))
    yaml_dir = OUT_DIR / "yaml"
    yaml_dir.mkdir(parents=True, exist_ok=True)
    for i, fold in enumerate(folds):
        val = set(fold)
        (yaml_dir / f"fold{i}_train.txt").write_text("".join(f"{p}\n" for s, p in sorted(images.items()) if s not in val))
        (yaml_dir / f"fold{i}_val.txt").write_text("".join(f"{p}\n" for s, p in sorted(images.items()) if s in val))
        data = {"train": str(yaml_dir / f"fold{i}_train.txt"), "val": str(yaml_dir / f"fold{i}_val.txt"), "names": NAMES}
        (yaml_dir / f"fold{i}.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))

    sizes = ", ".join(f"fold{i}={len(fold)} 張/{len({info[s]['patient'] for s in fold})} 人" for i, fold in enumerate(folds))
    print(f"病人層級 {K_FOLDS} 折切分: 共 {len(images)} 張、{len(patients)} 位病人; {sizes} (split seed={SPLIT_SEED})")


def yolo_dir(fold):
    return ROOT / "runs" / "segment" / PROJECT / f"yolo_fold{fold}"


def save_masks(masks, path):
    # 兩個類別可能重疊,以位元旗標存成一張圖: 1 = mf, 2 = ian, 3 = 兩者皆是
    image = np.zeros(next(iter(masks.values())).shape, dtype=np.uint8)
    for c, cls in NAMES.items():
        image[masks[cls]] |= 1 << c
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def metric_rows(dataset, fold, stem, tag, masks, gt):
    save_masks(masks, OUT_DIR / "masks" / dataset / tag / f"fold{fold}" / f"{stem}.png")
    return [
        {"dataset": dataset, "fold": fold, "filename": stem, "variant": tag, "class": cls, **pixel_metrics(masks[cls], gt[cls])}
        for cls in NAMES.values()
    ]


def evaluate(dataset, fold, samples):
    # samples: [(影像路徑, 灰階影像, GT)]
    rows = []
    gts = {path.stem: gt for path, _, gt in samples}

    source = OUT_DIR / "yaml" / f"eval_{dataset}_fold{fold}.txt"
    source.write_text("".join(f"{path}\n" for path, _, _ in samples))
    yolo = YOLO(yolo_dir(fold) / "weights" / "best.pt")
    results = yolo.predict(source=str(source), conf=MIN_CONF, iou=IOU, max_det=50, retina_masks=True, stream=True, verbose=False)
    for result in results:
        stem = Path(result.path).stem
        shape = tuple(result.orig_shape)
        candidates = get_candidates(result, shape)
        masks = {cls: items[0][1] if items else np.zeros(shape, dtype=bool) for cls, items in candidates.items()}
        rows += metric_rows(dataset, fold, stem, YOLO_TAG, masks, gts[stem])
    del yolo

    config = CONFIGS[UNET_TAG]
    unet = build_model({**config, "pretrained": False}).to(DEVICE)
    unet.load_state_dict(torch.load(WEIGHTS_DIR / f"{UNET_TAG}_fold{fold}.pth", map_location=DEVICE)["model_state_dict"])
    unet.eval()
    for path, image, gt in samples:
        rows += metric_rows(dataset, fold, path.stem, UNET_TAG, predict_masks(unet, image, config["size"]), gt)
    del unet
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(K_FOLDS)))
    parser.add_argument("--images", default=IMAGES_DIR, help="獨立測試集的影像資料夾")
    parser.add_argument("--gt", default=GT_DIR, help="獨立測試集的標註資料夾")
    parser.add_argument("--check-only", action="store_true", help="只產生切分,不訓練")
    parser.add_argument("--eval-only", action="store_true", help="不訓練,只用已訓練好的折評分")
    args = parser.parse_args()

    # 輸出導向檔案(nohup)時預設會整塊緩衝,改成逐行寫出才能即時看到進度
    sys.stdout.reconfigure(line_buffering=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if not PATIENT_MAP.exists():
        raise SystemExit(f"找不到 {PATIENT_MAP},請先執行 recover_patient_ids.py")

    build_folds(collect_images((ROOT / args.dataset).resolve()))
    if args.check_only:
        return

    if not args.eval_only:
        # 預訓練權重下載失敗時,在開始訓練前就會發現
        build_model(CONFIGS[UNET_TAG])
        for fold in args.folds:
            # 訓練結束才會產生 results.png / .done;已訓練完的折直接沿用
            if (yolo_dir(fold) / "results.png").exists():
                print(f"[已完成] yolo fold={fold}")
            else:
                model = YOLO(YOLO_MODEL)
                data_yaml = OUT_DIR / "yaml" / f"fold{fold}.yaml"
                model.train(data=str(data_yaml), project=PROJECT, name=f"yolo_fold{fold}", seed=SEED, save=True, plots=True, exist_ok=True, **YOLO_ARGS)
                del model
            gc.collect()
            torch.cuda.empty_cache()

            if (WEIGHTS_DIR / f"{UNET_TAG}_fold{fold}.done").exists():
                print(f"[已完成] {UNET_TAG} fold={fold}")
            else:
                train(UNET_TAG, CONFIGS[UNET_TAG], fold, UNET_ARGS["epochs"], OUT_DIR, WEIGHTS_DIR)
            gc.collect()
            torch.cuda.empty_cache()

    done = [
        fold
        for fold in range(K_FOLDS)
        if (yolo_dir(fold) / "results.png").exists() and (WEIGHTS_DIR / f"{UNET_TAG}_fold{fold}.done").exists()
    ]
    if not done:
        raise SystemExit("沒有任何兩個模型都訓練完成的折")

    test_gt = (ROOT / args.gt).resolve()
    test_samples = []
    for path in index_files((ROOT / args.images).resolve()).values():
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        test_samples.append((path, image, load_gt(path.stem, image.shape, test_gt, {})))

    for dataset in ("cv", "test"):
        rows = []
        for fold in done:
            samples = load_samples(fold_paths(fold, "val", OUT_DIR)) if dataset == "cv" else test_samples
            rows += evaluate(dataset, fold, samples)
            print(f"[{dataset}] fold {fold}: {len(samples)} 張")
        per_image_csv = OUT_DIR / f"compare_{dataset}.csv"
        with open(per_image_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        summarize(dataset, per_image_csv, YOLO_TAG)

    print(f"\n各折的預測遮罩已存到: {OUT_DIR / 'masks'}")
    if len(done) < K_FOLDS:
        print(f"[注意] 目前只包含 {done} 這幾折,請以五折都完成後的結果為準")


if __name__ == "__main__":
    main()
