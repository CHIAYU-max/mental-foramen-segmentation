import argparse
import csv
import gc
import statistics
import sys

import cv2
import torch

from compare_yolo_unet import GT_DIR, IMAGES_DIR, NAMES, ROOT, index_files, load_gt
from kfold_patient import OUT_DIR as PATIENT_DIR
from kfold_patient import PATIENT_MAP, UNET_TAG, save_masks
from kfold_unet_variants import CONFIGS, build_model, fold_paths, predict_masks, train
from kfold_yolo_unet import K_FOLDS, UNET_ARGS, load_samples
from paper_metrics import collect, image_metrics, read_masks
from predict_unet import DEVICE

# 分辨「第二批本身比較難」還是「兩批的標註標準不同」:
# 沿用 kfold_patient.py 的病人層級切分,每一折只用其中一批的訓練影像訓練最終設定的 U-Net,
# 再分別在兩批的驗證影像上評分,得到「訓練批次 × 評分批次」的 2×2 結果。
#   - 第二批訓練、第二批評分仍然很差:第二批本身難,或它內部的標註不一致
#   - 各自在自己那一批表現好、換到另一批就變差:兩批的標註定義不同
# 每次訓練只用約一半的資料,學習曲線顯示這大約會讓分數低 0.02~0.03,比較時要考慮進去。
CONFIG = CONFIGS[UNET_TAG]
BATCHES = ["first", "second"]
OUT_DIR = ROOT / "compare_results" / "cross_batch_mf_559"
WEIGHTS_DIR = OUT_DIR / "weights"


def variant(batch):
    return f"train_{batch}"


def write_lists(batch_of):
    for batch in BATCHES:
        yaml_dir = OUT_DIR / variant(batch) / "yaml"
        yaml_dir.mkdir(parents=True, exist_ok=True)
        for fold in range(K_FOLDS):
            paths = [p for p in fold_paths(fold, "train", PATIENT_DIR) if batch_of[p.stem] == batch]
            (yaml_dir / f"fold{fold}_train.txt").write_text("".join(f"{p}\n" for p in paths))
            (yaml_dir / f"fold{fold}_val.txt").write_text((PATIENT_DIR / "yaml" / f"fold{fold}_val.txt").read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(K_FOLDS)))
    parser.add_argument("--eval-only", action="store_true", help="不訓練,只用已訓練好的折評分")
    args = parser.parse_args()

    # 輸出導向檔案(nohup)時預設會整塊緩衝,改成逐行寫出才能即時看到進度
    sys.stdout.reconfigure(line_buffering=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(PATIENT_MAP, newline="") as f:
        batch_of = {r["stem"]: r["batch"] for r in csv.DictReader(f)}
    write_lists(batch_of)

    if not args.eval_only:
        build_model(CONFIG)
        for fold in args.folds:
            for batch in BATCHES:
                name = variant(batch)
                if (WEIGHTS_DIR / f"{name}_fold{fold}.done").exists():
                    print(f"[已完成] {name} fold={fold}")
                    continue
                train(name, CONFIG, fold, UNET_ARGS["epochs"], OUT_DIR / name, WEIGHTS_DIR)
                gc.collect()
                torch.cuda.empty_cache()

    # 預測並存下遮罩,再用與 paper_metrics.py 相同的方式計算指標
    test_gt = (ROOT / GT_DIR).resolve()
    test_samples = []
    for stem, path in index_files((ROOT / IMAGES_DIR).resolve()).items():
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        test_samples.append((path, image, load_gt(stem, image.shape, test_gt, {})))

    cv_samples, cv_batches = [], []
    done = {}
    for batch in BATCHES:
        name = variant(batch)
        done[batch] = [f for f in range(K_FOLDS) if (WEIGHTS_DIR / f"{name}_fold{f}.done").exists()]
    folds = sorted(set(done["first"]) & set(done["second"]))
    if not folds:
        raise SystemExit("還沒有兩批都訓練完成的折")
    for fold in folds:
        val = load_samples(fold_paths(fold, "val", PATIENT_DIR))
        for batch in BATCHES:
            name = variant(batch)
            model = build_model({**CONFIG, "pretrained": False}).to(DEVICE)
            model.load_state_dict(torch.load(WEIGHTS_DIR / f"{name}_fold{fold}.pth", map_location=DEVICE)["model_state_dict"])
            model.eval()
            for dataset, samples in (("cv", val), ("test", test_samples)):
                for path, image, _ in samples:
                    save_masks(predict_masks(model, image, CONFIG["size"]), OUT_DIR / "masks" / dataset / name / f"fold{fold}" / f"{path.stem}.png")
            del model
        cv_samples += [(fold, path.stem, gt) for path, _, gt in val]
        cv_batches += [batch_of[path.stem] for path, _, _ in val]

    # 全部資料訓練的最終模型作為對照
    sources = [("全部資料", PATIENT_DIR, UNET_TAG)] + [(f"只用 {b}", OUT_DIR, variant(b)) for b in BATCHES]
    print(f"\n使用的折: {folds}(交叉驗證只含這幾折的驗證影像;測試集為這幾折模型的多數決)")
    print(f"{'訓練資料':<14}{'評分資料':<14}{'n':>5}{'mf Dice':>10}{'命中 1R':>10}{'ian Dice':>10}")
    for label, root, name in sources:
        rows = collect(root, name, "cv", cv_samples, [], False)
        for batch in BATCHES:
            sub = [r for r, b in zip(rows, cv_batches) if b == batch]
            hits = [r["mf_hit_1R"] for r in sub if "mf_hit_1R" in r]
            print(f"{label:<14}{'cv / ' + batch:<14}{len(sub):>5}{statistics.mean(r['mf_dice'] for r in sub):>10.4f}{statistics.mean(hits):>10.1%}{statistics.mean(r['ian_dice'] for r in sub):>10.4f}")
        # 測試集用這幾折的多數決;全部資料的模型改用相同的折,才能公平比較
        metrics = []
        for path, _, gt in test_samples:
            preds = [read_masks(root, "test", name, fold, path.stem) for fold in folds]
            vote = {cls: sum(p[cls].astype(int) for p in preds) > len(folds) / 2 for cls in NAMES.values()}
            metrics.append(image_metrics(vote, gt))
        hits = [m["mf_hit_1R"] for m in metrics if "mf_hit_1R" in m]
        print(f"{label:<14}{'test':<14}{len(metrics):>5}{statistics.mean(m['mf_dice'] for m in metrics):>10.4f}{statistics.mean(hits):>10.1%}{statistics.mean(m['ian_dice'] for m in metrics):>10.4f}")
    print("\n[注意] 只用單一批次訓練時資料量約為一半,學習曲線顯示這大約會低 0.02~0.03")


if __name__ == "__main__":
    main()
