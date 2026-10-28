import argparse
import csv
import gc
import re
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from compare_yolo_unet import GT_DIR, IMAGES_DIR, NAMES, ROOT, index_files, load_gt, pixel_metrics
from eval_kfold_variants import FIELDS, summarize
from kfold_patient import OUT_DIR as PATIENT_DIR
from kfold_patient import PATIENT_MAP, UNET_TAG
from kfold_unet_variants import CONFIGS, DECODER_CHANNELS, ResNetUNet, augment, fold_paths, save_best
from kfold_yolo_unet import K_FOLDS, SEED, UNET_ARGS, dice_loss, load_samples
from predict_unet import DEVICE, DoubleConv

# 在 kfold_patient.py 的病人層級切分上,以目前最好的 U-Net(ResNet34 預訓練 + 強增強)為基礎,
# 一次只改一個設計,看能否再提升:
#   canonical: 把右側影像鏡射成左側的方向,訓練時不再隨機左右翻轉,讓模型學到固定的位置關係
#   coord:     在輸出層前加入座標通道,讓模型可以直接利用「目標通常出現在 ROI 的哪個位置」
#   mf_only:   只訓練頦孔一個類別
VARIANTS = {
    "canonical": dict(canonical=True, coord=False, classes=["mf", "ian"]),
    "coord": dict(canonical=False, coord=True, classes=["mf", "ian"]),
    "canonical_coord": dict(canonical=True, coord=True, classes=["mf", "ian"]),
    "mf_only": dict(canonical=False, coord=False, classes=["mf"]),
}
BASE = CONFIGS[UNET_TAG]
SIDE = re.compile(r"_(Left|Right)_roi$")

OUT_DIR = ROOT / "compare_results" / "kfold_patient_variants_mf_559"
WEIGHTS_DIR = OUT_DIR / "weights"


class Model(ResNetUNet):
    def __init__(self, num_classes, coord, pretrained):
        super().__init__(BASE["arch"], pretrained, num_classes)
        self.coord = coord
        if coord:
            self.head = nn.Sequential(DoubleConv(DECODER_CHANNELS[-1] + 2, 32, 32, bias=False), nn.Conv2d(32, num_classes, 1))

    def forward(self, x):
        size = x.shape[-2:]
        x = (x.repeat(1, 3, 1, 1) - self.mean) / self.std
        features = [self.stem(x)]
        x = self.pool(features[0])
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        x = features.pop()
        for decoder in self.decoders:
            skip = features.pop()
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([x, skip], dim=1))
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        if self.coord:
            # 範圍 -1~1 的垂直與水平座標
            ys = torch.linspace(-1, 1, size[0], device=x.device, dtype=x.dtype)
            xs = torch.linspace(-1, 1, size[1], device=x.device, dtype=x.dtype)
            grid = torch.stack(torch.meshgrid(ys, xs, indexing="ij")).expand(len(x), -1, -1, -1)
            x = torch.cat([x, grid], dim=1)
        return self.head(x)


def orient(samples, sides, canonical):
    # canonical 時把右側影像與標註一起左右鏡射;Dice 不受鏡射影響,評分時不需要轉回來
    if not canonical:
        return samples
    oriented = []
    for path, image, gt in samples:
        if sides[path.stem] == "Right":
            image = np.ascontiguousarray(image[:, ::-1])
            gt = {cls: np.ascontiguousarray(m[:, ::-1]) for cls, m in gt.items()}
        oriented.append((path, image, gt))
    return oriented


@torch.inference_mode()
def predict_masks(model, image, classes):
    h, w = image.shape
    size = BASE["size"]
    x = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    logits = model(torch.from_numpy(x)[None, None].to(DEVICE))
    pred = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False).argmax(1)[0].cpu().numpy()
    return {cls: pred == i + 1 for i, cls in enumerate(classes)}


def to_tensors(samples, classes):
    size = BASE["size"]
    xs, ys = [], []
    for _, image, gt in samples:
        mask = np.zeros(image.shape, dtype=np.uint8)
        for i, cls in enumerate(classes):
            mask[gt[cls]] = i + 1
        xs.append(cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_AREA))
        ys.append(cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST))
    x = torch.from_numpy(np.stack(xs)).float().div(255).unsqueeze(1)
    y = torch.from_numpy(np.stack(ys)).long()
    return x, y


def val_dice(model, samples, classes):
    model.eval()
    scores = []
    for _, image, gt in samples:
        masks = predict_masks(model, image, classes)
        scores += [pixel_metrics(masks[cls], gt[cls])["dice"] for cls in classes]
    return sum(scores) / len(scores)


def train(name, variant, fold, sides):
    torch.manual_seed(SEED)
    classes = variant["classes"]
    train_x, train_y = to_tensors(orient(load_samples(fold_paths(fold, "train", PATIENT_DIR)), sides, variant["canonical"]), classes)
    val_samples = orient(load_samples(fold_paths(fold, "val", PATIENT_DIR)), sides, variant["canonical"])

    model = Model(len(classes) + 1, variant["coord"], pretrained=True).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE["lr"], weight_decay=UNET_ARGS["weight_decay"])
    epochs = UNET_ARGS["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    best_path = WEIGHTS_DIR / f"{name}_fold{fold}.pth"

    # 與其他模型相同,以驗證集表現最好的 epoch 作為最終模型
    best_dice = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_x))
        losses = []
        for i in range(0, len(order), UNET_ARGS["batch"]):
            idx = order[i : i + UNET_ARGS["batch"]]
            if len(idx) < 2:
                continue
            # 方向已統一時不能再隨機左右翻轉
            x, y = augment(train_x[idx].to(DEVICE), train_y[idx].to(DEVICE), True, hflip=not variant["canonical"])
            with torch.autocast(DEVICE, enabled=use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits, y) + dice_loss(logits, y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        scheduler.step()

        dice = val_dice(model, val_samples, classes)
        if dice > best_dice and save_best({"epoch": epoch, "model_state_dict": model.state_dict(), "mean_dice": dice}, best_path):
            best_dice = dice
        print(f"[{name} fold{fold}] epoch {epoch}/{epochs} loss {sum(losses) / len(losses):.4f} val dice {dice:.4f} (best {best_dice:.4f})")

    # 訓練完整跑完且有成功存下權重才留下標記,否則下次會重新訓練
    if best_dice >= 0:
        best_path.with_suffix(".done").touch()
    del model


def evaluate(name, variant, folds, dataset, test_samples, sides):
    rows = []
    for fold in folds:
        model = Model(len(variant["classes"]) + 1, variant["coord"], pretrained=False).to(DEVICE)
        model.load_state_dict(torch.load(WEIGHTS_DIR / f"{name}_fold{fold}.pth", map_location=DEVICE)["model_state_dict"])
        model.eval()
        samples = load_samples(fold_paths(fold, "val", PATIENT_DIR)) if dataset == "cv" else test_samples
        for path, image, gt in orient(samples, sides, variant["canonical"]):
            masks = predict_masks(model, image, variant["classes"])
            for cls in variant["classes"]:
                row = {"dataset": dataset, "fold": fold, "filename": path.stem, "variant": name, "class": cls}
                rows.append({**row, **pixel_metrics(masks[cls], gt[cls])})
        del model
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
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

    # 訓練資料的左右側來自病人對照表,測試集則直接由檔名判斷
    test_gt = (ROOT / args.gt).resolve()
    test_samples = []
    with open(PATIENT_MAP, newline="") as f:
        sides = {r["stem"]: r["side"] for r in csv.DictReader(f)}
    for path in index_files((ROOT / args.images).resolve()).values():
        if not SIDE.search(path.stem):
            raise SystemExit(f"無法由檔名判斷左右側: {path.name}")
        sides[path.stem] = SIDE.search(path.stem)[1]
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        test_samples.append((path, image, load_gt(path.stem, image.shape, test_gt, {})))

    if not args.eval_only:
        # 先跑完一整折的所有版本再換下一折,中途的結果也能互相比較
        for fold in args.folds:
            for name in args.variants:
                if (WEIGHTS_DIR / f"{name}_fold{fold}.done").exists():
                    print(f"[已完成] {name} fold={fold}")
                    continue
                train(name, VARIANTS[name], fold, sides)
                gc.collect()
                torch.cuda.empty_cache()

    for dataset in ("cv", "test"):
        # kfold_patient.py 的結果作為比較基準
        with open(PATIENT_DIR / f"compare_{dataset}.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        for name, variant in VARIANTS.items():
            folds = [f for f in range(K_FOLDS) if (WEIGHTS_DIR / f"{name}_fold{f}.done").exists()]
            if folds:
                rows += evaluate(name, variant, folds, dataset, test_samples, sides)
                print(f"[{dataset}] {name}: 已完成的折 {folds}")
        merged_csv = OUT_DIR / f"compare_{dataset}.csv"
        with open(merged_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        summarize(dataset, merged_csv, UNET_TAG)
    print("\n[注意] 尚未跑完五折的版本只包含已完成的折,和其他版本的比較基礎不同,請以五折都完成後的結果為準")


if __name__ == "__main__":
    main()
