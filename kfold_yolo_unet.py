import argparse
import csv
import gc
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from ultralytics import YOLO

from compare_preprocessing import collect_images
from compare_yolo_unet import CONF, FIELDS, IOU, MODELS, NAMES, ROOT, load_gt, pixel_metrics, summarize, yolo_masks
from predict_unet import DEVICE, DoubleConv, predict

# YOLO 與 U-Net 共用的資料集(底下所有 images/ 內的影像都會納入,標註在同層的 labels/)
DATASET = "datasets/20260909/mf_559__unsharp_mask"
PROJECT = "kfold_yolo_unet_mf_559"
# 兩個模型共用同一份 K 折切分:依檔名(不含副檔名)決定每張影像屬於哪一折
K_FOLDS = 5
SPLIT_SEED = 0
SEED = 0

YOLO_MODEL = "yolov8n-seg.pt"
YOLO_ARGS = dict(epochs=200, patience=50, imgsz=1280, batch=16, device=0, workers=8)
# 輸入尺寸與正規化沿用 unet_predict.py;原始訓練程式已遺失,其餘超參數為重新設定
UNET_SIZE = 512
UNET_NORM = "div255"
UNET_ARGS = dict(epochs=100, batch=8, lr=1e-3, weight_decay=1e-4)

OUT_DIR = ROOT / "compare_results" / PROJECT
PER_IMAGE_CSV = OUT_DIR / "per_image.csv"
SUMMARY_CSV = OUT_DIR / "summary.csv"
KFOLD_FIELDS = ["fold"] + FIELDS


class UNet(nn.Module):
    # 與 best_unet(unsharp_mask).pth 相同的結構與層名稱,存下的權重可直接給 predict_unet.py 使用
    stride = 16

    def __init__(self, num_classes):
        super().__init__()
        self.enc1 = DoubleConv(1, 64, 64, bias=False)
        self.enc2 = DoubleConv(64, 128, 128, bias=False)
        self.enc3 = DoubleConv(128, 256, 256, bias=False)
        self.enc4 = DoubleConv(256, 512, 512, bias=False)
        self.bottleneck = DoubleConv(512, 1024, 1024, bias=False)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512, 512, bias=False)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256, 256, bias=False)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128, 128, bias=False)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64, 64, bias=False)
        self.out_conv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        b = self.bottleneck(F.max_pool2d(e4, 2))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


def gt_for(path, shape):
    return load_gt(path.stem, shape, path.parent.parent / "labels", {})


def metric_rows(fold, stem, model_name, masks, gt):
    return [
        {"fold": fold, "filename": stem, "model": model_name, "class": cls, **pixel_metrics(masks[cls], gt[cls])}
        for cls in NAMES.values()
    ]


def build_folds(images):
    stems = sorted(images)
    random.Random(SPLIT_SEED).shuffle(stems)
    folds = [set(stems[i::K_FOLDS]) for i in range(K_FOLDS)]
    with open(OUT_DIR / "folds.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stem", "fold"])
        writer.writerows(sorted((s, i) for i, fold in enumerate(folds) for s in fold))
    sizes = ", ".join(f"fold{i}={len(fold)}" for i, fold in enumerate(folds))
    print(f"{K_FOLDS} 折切分: 共 {len(stems)} 張, 各折驗證集 {sizes} (split seed={SPLIT_SEED})")
    return folds


def read_rows():
    if not PER_IMAGE_CSV.exists():
        return []
    with open(PER_IMAGE_CSV, newline="") as f:
        return list(csv.DictReader(f))


def append_rows(rows):
    new_file = not PER_IMAGE_CSV.exists()
    with open(PER_IMAGE_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=KFOLD_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def run_yolo(fold, train_paths, val_paths):
    yaml_dir = OUT_DIR / "yaml"
    yaml_dir.mkdir(parents=True, exist_ok=True)
    train_txt = yaml_dir / f"fold{fold}_train.txt"
    val_txt = yaml_dir / f"fold{fold}_val.txt"
    train_txt.write_text("".join(f"{p}\n" for p in train_paths))
    val_txt.write_text("".join(f"{p}\n" for p in val_paths))
    data_yaml = yaml_dir / f"fold{fold}.yaml"
    data = {"train": str(train_txt), "val": str(val_txt), "names": NAMES}
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))

    # 訓練結束才會產生 results.png;已訓練完的折直接沿用權重,只重做預測
    save_dir = ROOT / "runs" / "segment" / PROJECT / f"yolo_fold{fold}"
    if not (save_dir / "results.png").exists():
        model = YOLO(YOLO_MODEL)
        model.train(data=str(data_yaml), project=PROJECT, name=f"yolo_fold{fold}", seed=SEED, save=True, plots=True, **YOLO_ARGS)
        save_dir = Path(model.trainer.save_dir)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # 與 compare_yolo_unet.py 相同的方式把預測轉成原圖解析度的遮罩
    # source 用影像清單檔:直接傳路徑 list 會被當成一整個 batch 一次推論,且 result.path 不會是原檔名
    best = YOLO(save_dir / "weights" / "best.pt")
    rows = []
    results = best.predict(source=str(val_txt), conf=CONF, iou=IOU, retina_masks=True, stream=True, verbose=False)
    for result in results:
        path = Path(result.path)
        shape = tuple(result.orig_shape)
        rows += metric_rows(fold, path.stem, "yolo", yolo_masks(result, shape), gt_for(path, shape))
    del best
    return rows


def load_samples(paths):
    samples = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise SystemExit(f"無法讀取影像: {path}")
        samples.append((path, image, gt_for(path, image.shape)))
    return samples


def to_tensors(samples):
    # 縮放到訓練尺寸;遮罩為類別索引圖(0 背景, 1 mf, 2 ian)
    xs, ys = [], []
    for _, image, gt in samples:
        mask = np.zeros(image.shape, dtype=np.uint8)
        for c, cls in NAMES.items():
            mask[gt[cls]] = c + 1
        xs.append(cv2.resize(image, (UNET_SIZE, UNET_SIZE), interpolation=cv2.INTER_AREA))
        ys.append(cv2.resize(mask, (UNET_SIZE, UNET_SIZE), interpolation=cv2.INTER_NEAREST))
    x = torch.from_numpy(np.stack(xs)).float().div(255).unsqueeze(1)
    y = torch.from_numpy(np.stack(ys)).long()
    return x, y


def dice_loss(logits, target):
    probs = logits.float().softmax(dim=1)[:, 1:]
    onehot = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2)[:, 1:].float()
    inter = (probs * onehot).sum(dim=(0, 2, 3))
    total = probs.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))
    return 1 - ((2 * inter + 1) / (total + 1)).mean()


def predict_masks(model, image):
    pred = predict(model, image, UNET_SIZE, UNET_NORM)
    return {cls: pred == c + 1 for c, cls in NAMES.items()}


def val_dice(model, samples):
    model.eval()
    scores = []
    for _, image, gt in samples:
        masks = predict_masks(model, image)
        scores += [pixel_metrics(masks[cls], gt[cls])["dice"] for cls in NAMES.values()]
    return sum(scores) / len(scores)


def run_unet(fold, train_paths, val_paths):
    torch.manual_seed(SEED)
    train_x, train_y = to_tensors(load_samples(train_paths))
    val_samples = load_samples(val_paths)

    model = UNet(len(NAMES) + 1).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=UNET_ARGS["lr"], weight_decay=UNET_ARGS["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, UNET_ARGS["epochs"])
    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    weights_dir = OUT_DIR / "unet_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_path = weights_dir / f"unet_fold{fold}.pth"

    # 與 YOLO 的 best.pt 相同,以驗證集表現最好的 epoch 作為最終模型
    best_dice = -1.0
    for epoch in range(1, UNET_ARGS["epochs"] + 1):
        model.train()
        order = torch.randperm(len(train_x))
        losses = []
        for i in range(0, len(order), UNET_ARGS["batch"]):
            idx = order[i : i + UNET_ARGS["batch"]]
            if len(idx) < 2:
                continue
            x, y = train_x[idx].to(DEVICE), train_y[idx].to(DEVICE)
            # 增強: 左右翻轉、亮度與對比微調
            flip = torch.rand(len(idx), device=DEVICE) < 0.5
            x[flip], y[flip] = x[flip].flip(-1), y[flip].flip(-1)
            contrast = 1 + (torch.rand(len(idx), 1, 1, 1, device=DEVICE) - 0.5) * 0.4
            brightness = (torch.rand(len(idx), 1, 1, 1, device=DEVICE) - 0.5) * 0.2
            x = ((x - 0.5) * contrast + 0.5 + brightness).clamp(0, 1)
            with torch.autocast(DEVICE, enabled=use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits, y) + dice_loss(logits, y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        scheduler.step()

        dice = val_dice(model, val_samples)
        if dice > best_dice:
            best_dice = dice
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "mean_dice": dice}, best_path)
        print(f"[unet fold{fold}] epoch {epoch}/{UNET_ARGS['epochs']} loss {sum(losses) / len(losses):.4f} val dice {dice:.4f} (best {best_dice:.4f})")

    model.load_state_dict(torch.load(best_path, map_location=DEVICE)["model_state_dict"])
    model.eval()
    rows = []
    for path, image, gt in val_samples:
        rows += metric_rows(fold, path.stem, "unet", predict_masks(model, image), gt)
    del model
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(K_FOLDS)))
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--summary-only", action="store_true", help="只重新彙總已有結果,不訓練")
    parser.add_argument("--check-only", action="store_true", help="只檢查資料並產生切分,不訓練")
    args = parser.parse_args()

    # 輸出導向檔案(nohup)時預設會整塊緩衝,改成逐行寫出才能即時看到進度
    sys.stdout.reconfigure(line_buffering=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.summary_only:
        for fold in args.folds:
            if not 0 <= fold < K_FOLDS:
                parser.error(f"fold 必須介於 0 到 {K_FOLDS - 1}")
        images = collect_images((ROOT / args.dataset).resolve())
        folds = build_folds(images)
        if args.check_only:
            return

        # 已完成的 (fold, model) 不重跑,中斷後可直接重新執行接續
        done = {(int(r["fold"]), r["model"]) for r in read_rows()}
        runners = {"yolo": run_yolo, "unet": run_unet}
        for fold in args.folds:
            train_paths = [p for s, p in sorted(images.items()) if s not in folds[fold]]
            val_paths = [p for s, p in sorted(images.items()) if s in folds[fold]]
            for model_name in args.models:
                if (fold, model_name) in done:
                    print(f"[已完成] {model_name} fold={fold}")
                    continue
                append_rows(runners[model_name](fold, train_paths, val_paths))
                gc.collect()
                torch.cuda.empty_cache()

    summarize(PER_IMAGE_CSV, SUMMARY_CSV)


if __name__ == "__main__":
    main()
