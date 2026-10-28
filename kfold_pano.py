import argparse
import csv
import gc
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

from compare_preprocessing import collect_images
from compare_yolo_unet import GT_DIR, IMAGES_DIR, IOU, NAMES, ROOT, index_files, load_gt
from eval_kfold_variants import MIN_CONF
from kfold_patient import DATASET, PATIENT_MAP, UNET_TAG, save_masks
from kfold_unet_variants import CONFIGS, build_model, fold_paths, predict_masks, train
from kfold_yolo_unet import K_FOLDS, SEED, SPLIT_SEED, UNET_ARGS, YOLO_ARGS
from paper_metrics import image_metrics, read_masks
from predict_unet import DEVICE
from rerank_candidates import get_candidates

# 以「全景片」為單位的 K 折(2026-10-24):
#   kfold_patient.py 只依 patient_map.csv 的病人編號分折,但 locate_roi_sources.py 發現
#   同一張全景片的 ROI 可能有不同的病人編號,結果被分到不同折。
#   這裡把「同一位病人」與「同一張來源全景片」(roi_sources.csv 有比對到來源者,分數 ≥ 0.85)合併成同一組,
#   整組放進同一折;並把來源全景片也在訓練資料中的測試 ROI 從測試集拿掉。
# 超參數、訓練與推論規則(YOLO 各類別取信心最高的候選)與 kfold_patient.py 相同,只是可以換不同的 YOLO 模型。
PROJECT = "kfold_pano_mf_559"
ROI_SOURCES = ROOT / "compare_results" / "same_patient_check" / "roi_sources.csv"
# 209_Left_roi 的來源全景片 209.jpg 也在訓練資料中
EXCLUDE_TEST = {"209_Left_roi"}
UNET = "unet"
MODELS = [UNET, "yolov8n-seg.pt", "yolov8s-seg.pt", "yolo11n-seg.pt", "yolo11s-seg.pt", "yolo26n-seg.pt", "yolo26s-seg.pt"]
SUMMARY_KEYS = ["mf_dice", "mf_hit_1R", "ian_dice", "mf_predicted"]
# --batch 與 YOLO_ARGS 不同時,YOLO 的輸出名稱加上 _b<batch>(例如 yolo26s-seg_b8),不會與原設定的結果混在一起
TAG_SUFFIX = ""
# U-Net 用的設定(kfold_unet_variants.CONFIGS 的名稱);--unet-config 可換成別的設定,輸出名稱即設定名稱
UNET_CONFIG = UNET_TAG
# 彙總檔名的後綴: YOLO 改 batch 時為 _b<batch>,U-Net 換設定時為 _<設定名稱>
SUMMARY_SUFFIX = ""
# 訓練種子(2026-10-26 多種子重複): 與 SEED 不同時,U-Net 與 YOLO 的輸出名稱、彙總檔名都加 _s<種子>;切分仍用 SPLIT_SEED,不會改變
TRAIN_SEED = SEED
SEED_SUFFIX = ""


def tag_of(model):
    return (UNET_CONFIG if model == UNET else Path(model).stem + TAG_SUFFIX) + SEED_SUFFIX


def yolo_dir(project, tag, fold):
    return ROOT / "runs" / "segment" / project / f"{tag}_fold{fold}"


def is_done(model, fold, project, out_dir):
    # 訓練結束才會產生 results.png / .done;已訓練完的折直接沿用
    if model == UNET:
        return (out_dir / "weights" / f"{tag_of(UNET)}_fold{fold}.done").exists()
    return (yolo_dir(project, tag_of(model), fold) / "results.png").exists()


def build_groups(stems):
    # 同一位病人或同一張來源全景片的 ROI 以 union-find 合併成同一組
    with open(PATIENT_MAP, newline="") as f:
        info = {r["stem"]: r for r in csv.DictReader(f)}
    with open(ROI_SOURCES, newline="") as f:
        sources = {r["stem"]: r for r in csv.DictReader(f)}
    missing = sorted(set(stems) - set(info)) + sorted(set(stems) - set(sources))
    if missing:
        raise SystemExit(f"{len(missing)} 張影像不在 {PATIENT_MAP} 或 {ROI_SOURCES} 中(前5: {missing[:5]})")
    pano = {s: sources[s]["pano_group"] for s in stems if sources[s]["pano_group"]}

    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s in stems:
        parent[find(s)] = find(("patient", info[s]["patient"]))
        if s in pano:
            parent[find(s)] = find(("pano", pano[s]))
    groups = defaultdict(list)
    for s in sorted(stems):
        groups[find(s)].append(s)
    return list(groups.values()), info, pano, sources


def build_folds(images, out_dir, test_images):
    groups, info, pano, sources = build_groups(list(images))

    # 與 kfold_patient.py 相同: 兩批資料各自分配(跨兩批的組依多數歸類),每組放進目前影像數最少的那一折
    rng = random.Random(SPLIT_SEED)
    by_batch = defaultdict(list)
    for g in groups:
        by_batch[Counter(info[s]["batch"] for s in g).most_common(1)[0][0]].append(g)
    folds = [[] for _ in range(K_FOLDS)]
    for batch in sorted(by_batch):
        items = by_batch[batch]
        rng.shuffle(items)
        for g in sorted(items, key=lambda g: -len(g)):
            min(folds, key=len).extend(g)
    fold_of = {s: i for i, fold in enumerate(folds) for s in fold}
    group_of = {s: i for i, g in enumerate(groups) for s in g}

    # 檢查: 同一位病人、同一張全景片都只出現在一折
    for key, name in ((lambda s: info[s]["patient"], "病人"), (lambda s: pano.get(s), "全景片")):
        seen = defaultdict(set)
        for s in images:
            if key(s):
                seen[key(s)].add(fold_of[s])
        split = [k for k, v in seen.items() if len(v) > 1]
        if split:
            raise SystemExit(f"切分有誤: {len(split)} 個{name}跨折(前5: {split[:5]})")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "folds.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stem", "batch", "patient", "pano_group", "group", "fold"])
        writer.writerows(sorted((s, info[s]["batch"], info[s]["patient"], pano.get(s, ""), group_of[s], fold_of[s]) for s in images))
    yaml_dir = out_dir / "yaml"
    yaml_dir.mkdir(parents=True, exist_ok=True)
    for i, fold in enumerate(folds):
        val = set(fold)
        (yaml_dir / f"fold{i}_train.txt").write_text("".join(f"{p}\n" for s, p in sorted(images.items()) if s not in val))
        (yaml_dir / f"fold{i}_val.txt").write_text("".join(f"{p}\n" for s, p in sorted(images.items()) if s in val))
        data = {"train": str(yaml_dir / f"fold{i}_train.txt"), "val": str(yaml_dir / f"fold{i}_val.txt"), "names": NAMES}
        (yaml_dir / f"fold{i}.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    (yaml_dir / "test.txt").write_text("".join(f"{p}\n" for p in test_images.values()))

    sizes = Counter(len(g) for g in groups)
    print(f"全景片層級 {K_FOLDS} 折切分: 共 {len(images)} 張、{len(groups)} 組(病人 ∪ 來源全景片); 組大小分布 {dict(sorted(sizes.items()))} (split seed={SPLIT_SEED})")
    for i, fold in enumerate(folds):
        batches = Counter(info[s]["batch"] for s in fold)
        print(f"  fold{i}: {len(fold)} 張 / {len({group_of[s] for s in fold})} 組 / " + ", ".join(f"{b} {n}" for b, n in sorted(batches.items())))
    train_panos = set(pano.values())
    overlap = [s for s in test_images if sources.get(s, {}).get("pano_group") in train_panos]
    print(f"測試集: {len(test_images)} 張(已排除 {sorted(EXCLUDE_TEST)});來源全景片仍在訓練資料中的: {overlap or '無'}")


def train_fold(model, fold, args, out_dir):
    if model == UNET:
        train(tag_of(UNET), CONFIGS[UNET_CONFIG], fold, args.epochs or UNET_ARGS["epochs"], out_dir, out_dir / "weights", seed=TRAIN_SEED)
        return
    tag = tag_of(model)
    yolo = YOLO(model)
    train_args = {**YOLO_ARGS, "epochs": args.epochs or YOLO_ARGS["epochs"], "batch": args.batch or YOLO_ARGS["batch"]}
    yolo.train(data=str(out_dir / "yaml" / f"fold{fold}.yaml"), project=args.project, name=f"{tag}_fold{fold}", seed=TRAIN_SEED, save=True, plots=True, exist_ok=True, **train_args)
    save_dir = Path(yolo.trainer.save_dir)
    del yolo
    if save_dir != yolo_dir(args.project, tag, fold):
        raise SystemExit(f"YOLO 輸出到 {save_dir},與預期的 {yolo_dir(args.project, tag, fold)} 不同")
    # 磁碟空間有限: 評分只用 best.pt,last.pt 不保留
    (save_dir / "weights" / "last.pt").unlink(missing_ok=True)


def predict_fold(model, fold, project, out_dir, test_paths):
    # 預測遮罩存成位元旗標圖(與 kfold_patient.py 相同),之後的指標都從遮罩計算
    tag = tag_of(model)
    sets = (("cv", out_dir / "yaml" / f"fold{fold}_val.txt", fold_paths(fold, "val", out_dir)), ("test", out_dir / "yaml" / "test.txt", test_paths))
    if model == UNET:
        unet = build_model({**CONFIGS[UNET_CONFIG], "pretrained": False}).to(DEVICE)
        unet.load_state_dict(torch.load(out_dir / "weights" / f"{tag}_fold{fold}.pth", map_location=DEVICE)["model_state_dict"])
        unet.eval()
        for dataset, _, paths in sets:
            for path in paths:
                image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                save_masks(predict_masks(unet, image, CONFIGS[UNET_CONFIG]["size"]), out_dir / "masks" / dataset / tag / f"fold{fold}" / f"{path.stem}.png")
        del unet
        return
    yolo = YOLO(yolo_dir(project, tag, fold) / "weights" / "best.pt")
    for dataset, listing, _ in sets:
        for result in yolo.predict(source=str(listing), conf=MIN_CONF, iou=IOU, max_det=50, retina_masks=True, stream=True, verbose=False):
            shape = tuple(result.orig_shape)
            masks = {cls: items[0][1] if items else np.zeros(shape, dtype=bool) for cls, items in get_candidates(result, shape).items()}
            save_masks(masks, out_dir / "masks" / dataset / tag / f"fold{fold}" / f"{Path(result.path).stem}.png")
    del yolo


def mean(rows, key):
    values = [r[key] for r in rows if key in r and r[key] == r[key]]
    return statistics.mean(values) if values else float("nan")


def evaluate(tag, folds, out_dir, test):
    # 與 paper_metrics.py 相同: 交叉驗證每張影像只用沒看過它的那一折;測試集為單一模型平均與五折逐像素多數決
    cv = []
    for fold in folds:
        for path in fold_paths(fold, "val", out_dir):
            shape = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).shape
            gt = load_gt(path.stem, shape, path.parent.parent / "labels", {})
            cv.append(image_metrics(read_masks(out_dir, "cv", tag, fold, path.stem), gt))
    single, ensemble = [], []
    for stem, gt in test:
        preds = [read_masks(out_dir, "test", tag, fold, stem) for fold in folds]
        per_model = [image_metrics(p, gt) for p in preds]
        single.append({k: float(np.nanmean([m.get(k, np.nan) for m in per_model])) for k in per_model[0]})
        vote = {cls: np.sum([p[cls] for p in preds], axis=0) > len(folds) / 2 for cls in NAMES.values()}
        ensemble.append(image_metrics(vote, gt))
    return [
        {"model": tag, "setting": setting, "folds": " ".join(map(str, folds)), "n": len(rows), **{k: round(mean(rows, k), 4) for k in SUMMARY_KEYS}}
        for setting, rows in (("cv", cv), ("test single", single), ("test ensemble", ensemble))
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODELS, help=f"{UNET} 或 YOLO 權重檔名,依序訓練")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(K_FOLDS)))
    parser.add_argument("--epochs", type=int, default=None, help="測試腳本用;不給則用原本的設定(YOLO 200、U-Net 100)")
    parser.add_argument("--batch", type=int, default=None,
                        help="YOLO 的 batch;不給則用 YOLO_ARGS 的 16。2026-10-25 YOLO26s 在 batch 16 下與其他使用者共用 GPU 時一再 OOM 而改用 8"
                             "(Ultralytics 以 nbs=64 梯度累積,有效 batch、學習率與 weight decay 不變,只有 BatchNorm 每次看到的樣本數不同)")
    parser.add_argument("--project", default=PROJECT, help="輸出資料夾名稱;測試腳本時請換一個名稱")
    parser.add_argument("--check-only", action="store_true", help="只產生切分,不訓練")
    parser.add_argument("--eval-only", action="store_true", help="不訓練,只用已訓練好的折評分")
    parser.add_argument("--unet-config", default=UNET_TAG, choices=list(CONFIGS), help="U-Net 的設定,預設與主結果相同")
    parser.add_argument("--seed", type=int, default=SEED, help=f"訓練種子,預設 {SEED}(主結果);切分不受影響")
    args = parser.parse_args()
    global TAG_SUFFIX, UNET_CONFIG, SUMMARY_SUFFIX, TRAIN_SEED, SEED_SUFFIX
    if args.batch and args.batch != YOLO_ARGS["batch"]:
        TAG_SUFFIX = f"_b{args.batch}"
    UNET_CONFIG = args.unet_config
    TRAIN_SEED = args.seed
    SEED_SUFFIX = f"_s{TRAIN_SEED}" if TRAIN_SEED != SEED else ""
    SUMMARY_SUFFIX = TAG_SUFFIX + (f"_{UNET_CONFIG}" if UNET_CONFIG != UNET_TAG else "") + SEED_SUFFIX

    # 輸出導向檔案時預設會整塊緩衝,改成逐行寫出才能即時看到進度
    sys.stdout.reconfigure(line_buffering=True)
    out_dir = ROOT / "compare_results" / args.project
    (out_dir / "weights").mkdir(parents=True, exist_ok=True)

    test_images = {s: p for s, p in index_files((ROOT / IMAGES_DIR).resolve()).items() if s not in EXCLUDE_TEST}
    build_folds(collect_images((ROOT / DATASET).resolve()), out_dir, test_images)
    if args.check_only:
        return

    if not args.eval_only:
        # 預訓練權重下載失敗時,在開始訓練前就會發現
        if UNET in args.models:
            build_model(CONFIGS[UNET_CONFIG])
        for model in args.models:
            for fold in args.folds:
                if is_done(model, fold, args.project, out_dir):
                    print(f"[已完成] {tag_of(model)} fold={fold}")
                    continue
                print(f"[開始訓練] {tag_of(model)} fold={fold}")
                train_fold(model, fold, args, out_dir)
                gc.collect()
                torch.cuda.empty_cache()

    test_gt = (ROOT / GT_DIR).resolve()
    test = []
    for stem, path in test_images.items():
        shape = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).shape
        test.append((stem, load_gt(stem, shape, test_gt, {})))

    summary = []
    for model in args.models:
        folds = [f for f in args.folds if is_done(model, f, args.project, out_dir)]
        if not folds:
            print(f"[略過] {tag_of(model)}: 沒有訓練完成的折")
            continue
        for fold in folds:
            predict_fold(model, fold, args.project, out_dir, list(test_images.values()))
        summary += evaluate(tag_of(model), folds, out_dir, test)
        gc.collect()
        torch.cuda.empty_cache()
    if not summary:
        raise SystemExit("沒有任何訓練完成的模型")

    print(f"\n{'model':<20}{'setting':<16}{'folds':<12}{'n':>5}" + "".join(f"{k:>14}" for k in SUMMARY_KEYS))
    for r in summary:
        print(f"{r['model']:<20}{r['setting']:<16}{r['folds']:<12}{r['n']:>5}" + "".join(f"{r[k]:>14.4f}" for k in SUMMARY_KEYS))
    with open(out_dir / f"summary{SUMMARY_SUFFIX}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"\n彙總結果: {out_dir / f'summary{SUMMARY_SUFFIX}.csv'};預測遮罩: {out_dir / 'masks'}")
    if any(r["folds"] != " ".join(map(str, range(K_FOLDS))) for r in summary):
        print("[注意] 有模型尚未跑完五折,請以五折都完成後的結果為準")


if __name__ == "__main__":
    main()
