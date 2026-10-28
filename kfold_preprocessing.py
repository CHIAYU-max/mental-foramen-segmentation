import argparse
import csv
import gc
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import friedmanchisquare, wilcoxon

from compare_preprocessing import collect_images
from compare_yolo_unet import GT_DIR, ROOT, index_files, load_gt
from kfold_patient import OUT_DIR as PATIENT_DIR
from kfold_patient import PATIENT_MAP, UNET_TAG, save_masks
from kfold_unet_variants import CONFIGS, augment, build_model, predict_masks, save_best, to_tensors, val_dice
from kfold_yolo_unet import K_FOLDS, SEED, UNET_ARGS, dice_loss
from paper_metrics import bootstrap, collect
from predict_unet import DEVICE

# 影像前處理的比較實驗。
#
# 設計(事先固定,不依結果調整):
#   - 所有版本從同一批「未處理」的原圖出發,讀取時才套用前處理;訓練、驗證、測試影像都套用相同的處理
#   - 影像、標註、病人層級五折切分、模型(ResNet34 U-Net)、訓練設定、隨機種子完全相同,唯一的差別是前處理
#   - 參數採用文獻與函式庫的常用預設值,不在本資料上調整,避免選擇偏差
#   - 兩種訓練設定:
#       strong: 最終模型的強增強(主要分析)
#       none:   不做任何資料增強(敏感度分析,檢查增強是否掩蓋了前處理的效果)
#   - 主要指標: 交叉驗證的逐張頦孔 Dice;次要指標: 頦孔命中率、神經管 Dice、測試集(五折多數決)
#   - 統計: Friedman 檢定(所有版本);各版本對 original 的逐張成對 Wilcoxon 檢定,以 Holm 法校正多重比較;
#           差異的 95% 信賴區間以病人為單位重抽樣
#   - 實務上等效的界限: ±0.02 Dice(與種子間、折間的波動同量級);差異的信賴區間完全落在界限內即視為等效
# 權重在預測完成後刪除以節省磁碟空間,預測遮罩會保留。
ORIGINAL_DIR = "dental_project/20260909/mf_559_original"
TEST_DIR = "dental_project/20261008/test"
SETTINGS = ["strong", "none"]
EQUIVALENCE = 0.02
OUT_DIR = ROOT / "compare_results" / "preprocessing_patient_mf_559"
WEIGHTS_DIR = OUT_DIR / "weights"


def gamma_lut(gamma):
    # 輸出 = 輸入^(1/gamma);gamma > 1 使影像變亮
    return np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).clip(0, 255).astype(np.uint8)


GAMMA_LUT = gamma_lut(1.5)
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def unsharp(image, sigma=1.0, amount=1.0):
    # 標準的 unsharp masking: 原圖 + amount × (原圖 − 高斯模糊),兩側邊緣都會加強
    image = image.astype(np.float32)
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return np.clip(image + amount * (image - blurred), 0, 255).astype(np.uint8)


def laplacian_sharpen(image, c=1.0):
    # Gonzalez & Woods 的拉普拉斯銳化: 原圖 − c × ∇²(原圖),4 鄰域核
    image = image.astype(np.float32)
    return np.clip(image - c * cv2.Laplacian(image, cv2.CV_32F, ksize=1), 0, 255).astype(np.uint8)


# 前處理與參數(灰階 uint8 → 灰階 uint8)
PREPROCESS = {
    "original": lambda img: img,
    "histogram_equalization": cv2.equalizeHist,
    "clahe": CLAHE.apply,
    "gamma": lambda img: cv2.LUT(img, GAMMA_LUT),
    "unsharp_mask": unsharp,
    "bilateral": lambda img: cv2.bilateralFilter(img, 9, 75, 75),
    "median": lambda img: cv2.medianBlur(img, 3),
    "gaussian": lambda img: cv2.GaussianBlur(img, (5, 5), 1.0),
    "laplacian": laplacian_sharpen,
    "bf_gamma_clahe_um": lambda img: unsharp(CLAHE.apply(cv2.LUT(cv2.bilateralFilter(img, 9, 75, 75), GAMMA_LUT))),
}
PARAMETERS = {
    "original": "無",
    "histogram_equalization": "全域直方圖等化",
    "clahe": "clipLimit 2.0、tile 8×8",
    "gamma": "γ = 1.5(輸出 = 輸入^(1/γ))",
    "unsharp_mask": "高斯 σ = 1.0、amount = 1.0",
    "bilateral": "d = 9、σcolor = 75、σspace = 75",
    "median": "核 3×3",
    "gaussian": "核 5×5、σ = 1.0",
    "laplacian": "4 鄰域拉普拉斯、c = 1",
    "bf_gamma_clahe_um": "bilateral → gamma → CLAHE → unsharp mask(參數同上)",
}


def name(setting, variant):
    return f"{setting}_{variant}"


def load_split(images, fold_of):
    # 回傳 {fold: [(路徑, 原圖, GT)]},原圖尚未前處理
    folds = {f: [] for f in range(K_FOLDS)}
    for stem, path in sorted(images.items()):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise SystemExit(f"無法讀取影像: {path}")
        folds[fold_of[stem]].append((path, image, load_gt(stem, image.shape, path.parent.parent / "labels", {})))
    return folds


def apply(samples, variant):
    return [(path, PREPROCESS[variant](image), gt) for path, image, gt in samples]


def train_one(train_samples, val_samples, setting, weights_path):
    config = CONFIGS[UNET_TAG]
    torch.manual_seed(SEED)
    train_x, train_y = to_tensors(train_samples, config["size"])
    model = build_model(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=UNET_ARGS["weight_decay"])
    epochs = UNET_ARGS["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    best = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_x))
        losses = []
        for i in range(0, len(order), UNET_ARGS["batch"]):
            idx = order[i : i + UNET_ARGS["batch"]]
            if len(idx) < 2:
                continue
            x, y = train_x[idx].to(DEVICE), train_y[idx].to(DEVICE)
            if setting == "strong":
                x, y = augment(x, y, True)
            with torch.autocast(DEVICE, enabled=use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits, y) + dice_loss(logits, y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        scheduler.step()
        dice = val_dice(model, val_samples, config["size"])
        if dice > best and save_best({"epoch": epoch, "model_state_dict": model.state_dict(), "mean_dice": dice}, weights_path):
            best = dice
        print(f"[{weights_path.stem}] epoch {epoch}/{epochs} loss {sum(losses) / len(losses):.4f} val dice {dice:.4f} (best {best:.4f})")
    del model
    return best >= 0


def predict_and_save(variant_name, fold, weights_path, val_samples, test_samples):
    config = CONFIGS[UNET_TAG]
    model = build_model({**config, "pretrained": False}).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE)["model_state_dict"])
    model.eval()
    for dataset, samples in (("cv", val_samples), ("test", test_samples)):
        for path, image, _ in samples:
            save_masks(predict_masks(model, image, config["size"]), OUT_DIR / "masks" / dataset / variant_name / f"fold{fold}" / f"{path.stem}.png")
    del model


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def load_visibility():
    # visibility_grading.py 的頦孔可辨識性判定;沒有的話回傳空字典
    path = ROOT / "visibility_grading" / "visibility.csv"
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {r["stem"]: int(r["mf_visible"]) for r in csv.DictReader(f)}


def analyze(setting, variants, cv_samples, cv_patients, test_samples, test_patients, writer):
    rng = np.random.default_rng(0)
    visibility = load_visibility()
    for dataset, samples, patients, label in (
        ("cv", cv_samples, cv_patients, "交叉驗證"),
        ("test", test_samples, test_patients, "測試集(五折多數決)"),
    ):
        results = {}
        for variant in variants:
            rows = collect(OUT_DIR, name(setting, variant), dataset, cv_samples if dataset == "cv" else [], test_samples if dataset == "test" else [], dataset == "test")
            if rows is not None:
                results[variant] = rows
        if "original" not in results or len(results) < 2:
            print(f"\n[{setting} / {label}] 完成的版本不足,略過")
            continue

        dice = {v: np.array([r["mf_dice"] for r in rows]) for v, rows in results.items()}
        others = [v for v in results if v != "original"]
        friedman_p = friedmanchisquare(*dice.values()).pvalue if len(results) >= 3 else float("nan")
        raw_p = []
        for v in others:
            d = dice[v] - dice["original"]
            raw_p.append(float(wilcoxon(d).pvalue) if np.any(d) else 1.0)
        adjusted = dict(zip(others, holm(raw_p)))

        stems = [s for _, s, _ in cv_samples] if dataset == "cv" else [s for s, _ in test_samples]
        visible = np.array([visibility.get(s, -1) for s in stems])

        print(f"\n===== 訓練設定 {setting} / {label}:{len(results)} 個版本,Friedman p = {friedman_p:.3g} =====")
        print(f"{'前處理':<24}{'mf Dice':>9}{'可辨識':>8}{'無法辨識':>9}{'命中 1R':>9}{'ian Dice':>10}{'差異(vs original)':>20}{'95% CI':>20}{'Holm p':>10}  判定")
        for v, rows in results.items():
            hits = [r["mf_hit_1R"] for r in rows if "mf_hit_1R" in r]
            # 依頦孔可辨識性分開的平均 Dice(尚未判定時為 nan)
            vis = float(dice[v][visible == 1].mean()) if (visible == 1).any() else float("nan")
            invis = float(dice[v][visible == 0].mean()) if (visible == 0).any() else float("nan")
            text = f"{v:<24}{dice[v].mean():>9.4f}{vis:>8.4f}{invis:>9.4f}{np.mean(hits):>9.1%}{np.mean([r['ian_dice'] for r in rows]):>10.4f}"
            record = {"setting": setting, "dataset": dataset, "variant": v, "parameters": PARAMETERS[v], "n": len(rows),
                      "mf_dice": round(float(dice[v].mean()), 4), "mf_dice_visible": round(vis, 4), "mf_dice_not_visible": round(invis, 4),
                      "mf_hit_1R": round(float(np.mean(hits)), 4),
                      "ian_dice": round(float(np.mean([r["ian_dice"] for r in rows])), 4), "friedman_p": friedman_p}
            if v == "original":
                print(text + f"{'—':>20}{'—':>20}{'—':>10}  基準")
            else:
                diffs = [{"d": float(x)} for x in dice[v] - dice["original"]]
                est, low, high = bootstrap(diffs, patients, "d", np.mean, rng)
                if -EQUIVALENCE < low and high < EQUIVALENCE:
                    verdict = "實務上等效"
                elif adjusted[v] < 0.05:
                    verdict = "顯著較好" if est > 0 else "顯著較差"
                else:
                    verdict = "無法判定"
                print(text + f"{est:>+20.4f}{f'{low:+.4f} ~ {high:+.4f}':>20}{adjusted[v]:>10.3g}  {verdict}")
                record.update(diff=round(est, 4), ci_low=round(low, 4), ci_high=round(high, 4), holm_p=adjusted[v], verdict=verdict)
            writer.writerow(record)
    print(f"\n判定標準: 差異的 95% CI 完全落在 ±{EQUIVALENCE} 以內為「實務上等效」;否則 Holm 校正後 p < 0.05 為顯著;其餘為無法判定")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", nargs="+", default=SETTINGS, choices=SETTINGS)
    parser.add_argument("--variants", nargs="+", default=list(PREPROCESS), choices=list(PREPROCESS))
    parser.add_argument("--eval-only", action="store_true", help="不訓練,只分析已完成的結果")
    args = parser.parse_args()

    # 輸出導向檔案(nohup)時預設會整塊緩衝,改成逐行寫出才能即時看到進度
    sys.stdout.reconfigure(line_buffering=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(PATIENT_DIR / "folds.csv", newline="") as f:
        fold_of = {r["stem"]: int(r["fold"]) for r in csv.DictReader(f)}
    with open(PATIENT_MAP, newline="") as f:
        patient_of = {r["stem"]: r["patient"] for r in csv.DictReader(f)}
    images = collect_images((ROOT / ORIGINAL_DIR).resolve())
    missing = sorted(set(fold_of) - set(images))
    if missing:
        raise SystemExit(f"原圖資料夾缺少 {len(missing)} 張(前5: {missing[:5]}),請確認 ORIGINAL_DIR")
    images = {s: p for s, p in images.items() if s in fold_of}
    folds = load_split(images, fold_of)

    test_gt = (ROOT / GT_DIR).resolve()
    test_raw = []
    for stem, path in sorted(index_files((ROOT / TEST_DIR).resolve()).items()):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        test_raw.append((path, image, load_gt(stem, image.shape, test_gt, {})))
    print(f"訓練原圖 {len(images)} 張、測試原圖 {len(test_raw)} 張;前處理: {args.variants};訓練設定: {args.settings}")

    if not args.eval_only:
        build_model(CONFIGS[UNET_TAG])
        # 先完成主要分析(strong)的所有版本與折,再進行敏感度分析(none)
        for setting in args.settings:
            for fold in range(K_FOLDS):
                train_raw = [s for f in range(K_FOLDS) if f != fold for s in folds[f]]
                for variant in args.variants:
                    variant_name = name(setting, variant)
                    done = WEIGHTS_DIR / f"{variant_name}_fold{fold}.done"
                    if done.exists():
                        print(f"[已完成] {variant_name} fold={fold}")
                        continue
                    weights = WEIGHTS_DIR / f"{variant_name}_fold{fold}.pth"
                    val = apply(folds[fold], variant)
                    if train_one(apply(train_raw, variant), val, setting, weights):
                        predict_and_save(variant_name, fold, weights, val, apply(test_raw, variant))
                        done.touch()
                        weights.unlink(missing_ok=True)
                    gc.collect()
                    torch.cuda.empty_cache()

    cv_samples = [(fold, path.stem, gt) for fold in range(K_FOLDS) for path, _, gt in folds[fold]]
    cv_patients = [patient_of[stem] for _, stem, _ in cv_samples]
    test_samples = [(path.stem, gt) for path, _, gt in test_raw]
    test_patients = [path.stem.rsplit("_", 2)[0] for path, _, _ in test_raw]
    with open(OUT_DIR / "summary.csv", "w", newline="") as f:
        fields = ["setting", "dataset", "variant", "parameters", "n", "mf_dice", "mf_dice_visible", "mf_dice_not_visible", "mf_hit_1R", "ian_dice", "diff", "ci_low", "ci_high", "holm_p", "friedman_p", "verdict"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for setting in args.settings:
            variants = [v for v in PREPROCESS if all((WEIGHTS_DIR / f"{name(setting, v)}_fold{k}.done").exists() for k in range(K_FOLDS))]
            analyze(setting, variants, cv_samples, cv_patients, test_samples, test_patients, writer)
    print(f"\n彙總結果: {OUT_DIR / 'summary.csv'}(只納入五折都完成的版本)")


if __name__ == "__main__":
    main()
