import argparse
import csv
import gc
import sys
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from stage2.compare_yolo_unet import GT_DIR, IMAGES_DIR, IOU, NAMES, ROOT, index_files, load_gt, pixel_metrics
from stage2.eval_kfold_variants import FIELDS, MIN_CONF, summarize
from stage2.kfold_yolo_unet import K_FOLDS, SEED, YOLO_ARGS, YOLO_MODEL
from stage2.kfold_yolo_unet import OUT_DIR as BASE_DIR
from stage2.rerank_candidates import BASELINE as BASE_VARIANT
from stage2.rerank_candidates import csv_path as base_csv_path
from stage2.rerank_candidates import get_candidates

# 換一個 YOLO 模型,用 kfold_yolo_unet.py 的同一份 K 折切分重新訓練,
# 再以「各類別取信心最高的候選」評分,並與原本的模型逐張成對比較
MODEL = "yolov8s-seg.pt"
BASE_TAG = Path(YOLO_MODEL).stem


def predict_top1(weights, source, gt_dirs, dataset, fold, tag):
    yolo = YOLO(weights)
    results = yolo.predict(source=str(source), conf=MIN_CONF, iou=IOU, max_det=50, retina_masks=True, stream=True, verbose=False)
    rows = []
    for result in results:
        stem = Path(result.path).stem
        shape = tuple(result.orig_shape)
        gt = load_gt(stem, shape, gt_dirs[stem], {})
        candidates = get_candidates(result, shape)
        for cls in NAMES.values():
            mask = candidates[cls][0][1] if candidates[cls] else np.zeros(shape, dtype=bool)
            row = {"dataset": dataset, "fold": fold, "filename": stem, "variant": tag, "class": cls}
            rows.append({**row, **pixel_metrics(mask, gt[cls])})
    del yolo
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL, help="要比較的 YOLO 模型,例如 yolov8s-seg.pt 或 yolo11s-seg.pt")
    parser.add_argument("--batch", type=int, default=YOLO_ARGS["batch"], help="顯示卡記憶體不足時調小")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(K_FOLDS)))
    parser.add_argument("--images", default=IMAGES_DIR, help="獨立測試集的影像資料夾")
    parser.add_argument("--gt", default=GT_DIR, help="獨立測試集的標註資料夾")
    parser.add_argument("--eval-only", action="store_true", help="不訓練,只用已訓練好的折評分")
    args = parser.parse_args()

    # 輸出導向檔案(nohup)時預設會整塊緩衝,改成逐行寫出才能即時看到進度
    sys.stdout.reconfigure(line_buffering=True)

    tag = Path(args.model).stem
    if tag == BASE_TAG:
        parser.error(f"{args.model} 就是原本的模型,請指定不同的模型")
    project = f"kfold_{tag}_mf_559"
    out_dir = ROOT / "compare_results" / project
    (out_dir / "yaml").mkdir(parents=True, exist_ok=True)
    for dataset in ("cv", "test"):
        if not base_csv_path(dataset).exists():
            raise SystemExit(f"找不到原本模型的結果 {base_csv_path(dataset)},請先執行 rerank_candidates.py")

    # 直接沿用原本各折的資料清單,確保切分完全相同
    save_dirs = {}
    for fold in args.folds:
        data_yaml = BASE_DIR / "yaml" / f"fold{fold}.yaml"
        if not data_yaml.exists():
            raise SystemExit(f"找不到 {data_yaml},請先跑完 kfold_yolo_unet.py")
        # 訓練結束才會產生 results.png;已訓練完的折直接沿用
        save_dir = ROOT / "runs" / "segment" / project / f"yolo_fold{fold}"
        if not (save_dir / "results.png").exists():
            if args.eval_only:
                print(f"[未訓練] fold={fold},略過")
                continue
            model = YOLO(args.model)
            train_args = {**YOLO_ARGS, "batch": args.batch}
            model.train(data=str(data_yaml), project=project, name=f"yolo_fold{fold}", seed=SEED, save=True, plots=True, **train_args)
            save_dir = Path(model.trainer.save_dir)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        save_dirs[fold] = save_dir
    if not save_dirs:
        raise SystemExit("沒有任何已訓練完成的折")

    test_images = index_files((ROOT / args.images).resolve())
    test_gt = (ROOT / args.gt).resolve()
    for dataset in ("cv", "test"):
        rows = []
        for fold, save_dir in save_dirs.items():
            if dataset == "cv":
                source = BASE_DIR / "yaml" / f"fold{fold}_val.txt"
                paths = [Path(line) for line in source.read_text().splitlines() if line]
                gt_dirs = {p.stem: p.parent.parent / "labels" for p in paths}
            else:
                source = out_dir / "yaml" / "test.txt"
                source.write_text("".join(f"{p}\n" for p in test_images.values()))
                gt_dirs = {stem: test_gt for stem in test_images}
            rows += predict_top1(save_dir / "weights" / "best.pt", source, gt_dirs, dataset, fold, tag)
            print(f"[{dataset}] fold {fold}: {len(gt_dirs)} 張")

        # 原本模型在相同折上的結果,改用模型名稱標示後放在一起比較
        with open(base_csv_path(dataset), newline="") as f:
            base_rows = [r for r in csv.DictReader(f) if r["variant"] == BASE_VARIANT and int(r["fold"]) in save_dirs]
        for r in base_rows:
            r["variant"] = BASE_TAG

        merged_csv = out_dir / f"compare_{dataset}.csv"
        with open(merged_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(base_rows + rows)
        summarize(dataset, merged_csv, BASE_TAG)

    if len(save_dirs) < K_FOLDS:
        print(f"\n[注意] 目前只包含 {sorted(save_dirs)} 這幾折,兩個模型都只用這幾折比較")


if __name__ == "__main__":
    main()
