import argparse
import csv
import gc
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from stage2.compare_yolo_unet import GT_DIR, IMAGES_DIR, NAMES, ROOT, index_files, load_gt, pixel_metrics
from stage2.eval_kfold_variants import FIELDS, summarize
from stage2.eval_kfold_variants import csv_path as base_csv_path
from stage2.kfold_yolo_unet import K_FOLDS, SEED, UNET_ARGS, UNet, dice_loss, load_samples
from stage2.kfold_yolo_unet import OUT_DIR as BASE_DIR
from stage2.predict_unet import DEVICE, DoubleConv
from stage2.rerank_candidates import BASELINE as YOLO_VARIANT
from stage2.rerank_candidates import csv_path as yolo_csv_path

# 用 kfold_yolo_unet.py 的同一份 K 折切分,訓練不同預訓練與訓練設定的 U-Net,
# 再與原本從零訓練的 U-Net(unet_scratch)以及 YOLO 逐張成對比較
#   arch:       unet = 原本的結構(無預訓練可用);resnet34 / resnet50 = torchvision 的 encoder 加上 U-Net decoder
#   pretrained: encoder 是否載入 ImageNet 預訓練權重
#   aug:        basic = 左右翻轉 + 亮度對比;strong = 再加上縮放、旋轉、平移
#   size:       輸入尺寸 (高, 寬);(384, 512) 接近原圖的長寬比,不會把影像壓成正方形
CONFIGS = {
    "scratch_aug": dict(arch="unet", pretrained=False, aug="strong", size=(512, 512), lr=1e-3),
    "r34_imagenet": dict(arch="resnet34", pretrained=True, aug="basic", size=(512, 512), lr=3e-4),
    "r34_imagenet_aug": dict(arch="resnet34", pretrained=True, aug="strong", size=(512, 512), lr=3e-4),
    "r34_imagenet_aug_rect": dict(arch="resnet34", pretrained=True, aug="strong", size=(384, 512), lr=3e-4),
    "r50_imagenet_aug": dict(arch="resnet50", pretrained=True, aug="strong", size=(512, 512), lr=3e-4),
    # 2026-10-25: 端到端評估發現 r34_imagenet_aug 依賴「存過兩次 JPEG」的壓縮痕跡(記憶體內處理時命中率掉 2.2 個百分點),
    #   這個設定在 strong 增強後再隨機重新壓縮 JPEG,看能不能讓模型不依賴壓縮痕跡
    "r34_imagenet_aug_jpeg": dict(arch="resnet34", pretrained=True, aug="strong", size=(512, 512), lr=3e-4, jpeg=(0.5, 50, 95)),
    # 2026-10-27 精度對運算量曲線(SOP 用): 編碼器 ResNet18 / 34 × 輸入 384 / 512,其餘與 r34_imagenet_aug 相同
    "r18_imagenet_aug": dict(arch="resnet18", pretrained=True, aug="strong", size=(512, 512), lr=3e-4),
    "r34_imagenet_aug_384": dict(arch="resnet34", pretrained=True, aug="strong", size=(384, 384), lr=3e-4),
    "r18_imagenet_aug_384": dict(arch="resnet18", pretrained=True, aug="strong", size=(384, 384), lr=3e-4),
}
BASE_TAG = "unet_scratch"
ENCODER_CHANNELS = {"resnet18": [64, 64, 128, 256, 512], "resnet34": [64, 64, 128, 256, 512], "resnet50": [64, 256, 512, 1024, 2048]}
DECODER_CHANNELS = [256, 128, 64, 64]

OUT_DIR = ROOT / "compare_results" / "kfold_unet_variants_mf_559"
WEIGHTS_DIR = OUT_DIR / "weights"


class ResNetUNet(nn.Module):
    def __init__(self, arch, pretrained, num_classes):
        super().__init__()
        net = getattr(torchvision.models, arch)(weights="IMAGENET1K_V1" if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.pool = net.maxpool
        self.layers = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])
        # 輸入是 0~1 的單通道灰階,複製成三通道並套用 ImageNet 的正規化
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        skips = ENCODER_CHANNELS[arch][::-1]
        self.decoders = nn.ModuleList()
        c_in = skips[0]
        for c_skip, c_out in zip(skips[1:], DECODER_CHANNELS):
            self.decoders.append(DoubleConv(c_in + c_skip, c_out, c_out, bias=False))
            c_in = c_out
        self.head = nn.Sequential(DoubleConv(c_in, 32, 32, bias=False), nn.Conv2d(32, num_classes, 1))

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
        return self.head(F.interpolate(x, size=size, mode="bilinear", align_corners=False))


def build_model(config):
    if config["arch"] == "unet":
        return UNet(len(NAMES) + 1)
    return ResNetUNet(config["arch"], config["pretrained"], len(NAMES) + 1)


@torch.inference_mode()
def predict_masks(model, image, size):
    # 回傳原圖解析度的 {類別: bool 遮罩}
    h, w = image.shape
    x = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    logits = model(torch.from_numpy(x)[None, None].to(DEVICE))
    pred = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False).argmax(1)[0].cpu().numpy()
    return {cls: pred == c + 1 for c, cls in NAMES.items()}


def to_tensors(samples, size):
    xs, ys = [], []
    for _, image, gt in samples:
        mask = np.zeros(image.shape, dtype=np.uint8)
        for c, cls in NAMES.items():
            mask[gt[cls]] = c + 1
        xs.append(cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_AREA))
        ys.append(cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST))
    x = torch.from_numpy(np.stack(xs)).float().div(255).unsqueeze(1)
    y = torch.from_numpy(np.stack(ys)).long()
    return x, y


def augment(x, y, strong, hflip=True):
    n = len(x)
    if hflip:
        flip = torch.rand(n, device=x.device) < 0.5
        x[flip], y[flip] = x[flip].flip(-1), y[flip].flip(-1)
    if strong:
        # 縮放 0.8~1.2 倍、旋轉 ±15 度、平移 ±10%,影像與遮罩套用同一個變換
        angle = (torch.rand(n, device=x.device) - 0.5) * 2 * math.radians(15)
        scale = 0.8 + 0.4 * torch.rand(n, device=x.device)
        shift = (torch.rand(n, 2, device=x.device) - 0.5) * 0.4
        cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
        theta = torch.stack([torch.stack([cos, -sin, shift[:, 0]], dim=1), torch.stack([sin, cos, shift[:, 1]], dim=1)], dim=1)
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
        y = F.grid_sample(y[:, None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)[:, 0].long()
    contrast = 1 + (torch.rand(n, 1, 1, 1, device=x.device) - 0.5) * 0.4
    brightness = (torch.rand(n, 1, 1, 1, device=x.device) - 0.5) * 0.2
    return ((x - 0.5) * contrast + 0.5 + brightness).clamp(0, 1), y


def jpeg_augment(x, prob, q_min, q_max):
    # 每張影像以機率 prob 用隨機品質 [q_min, q_max] 壓成 JPEG 再解回(在 CPU 上做,batch 小所以很快)
    out = x.clone()
    for i in torch.nonzero(torch.rand(len(x)) < prob).flatten().tolist():
        img = (x[i, 0].float().cpu().numpy() * 255).round().astype(np.uint8)
        q = int(torch.randint(q_min, q_max + 1, (1,)))
        dec = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_GRAYSCALE)
        out[i, 0] = torch.from_numpy(dec).to(x.device, x.dtype) / 255
    return out


def val_dice(model, samples, size):
    model.eval()
    scores = []
    for _, image, gt in samples:
        masks = predict_masks(model, image, size)
        scores += [pixel_metrics(masks[cls], gt[cls])["dice"] for cls in NAMES.values()]
    return sum(scores) / len(scores)


def save_best(state, path):
    # 先寫到暫存檔再取代:磁碟空間不足而寫入失敗時,先前的最佳權重不會被寫壞,訓練也不會中斷
    tmp = path.with_suffix(".tmp")
    try:
        torch.save(state, tmp)
        tmp.replace(path)
        return True
    except (RuntimeError, OSError) as e:
        print(f"[警告] 權重寫入失敗(磁碟空間不足?),保留先前的最佳權重: {e}")
        tmp.unlink(missing_ok=True)
        return False


def fold_paths(fold, split, base_dir=BASE_DIR):
    return [Path(line) for line in (base_dir / "yaml" / f"fold{fold}_{split}.txt").read_text().splitlines() if line]


def train(name, config, fold, epochs, base_dir=BASE_DIR, weights_dir=None, seed=SEED, extra=None):
    # extra: 額外加進訓練集的影像(2026-10-28 半監督訓練的假標註,kfold_pseudo.py);標註讀影像上一層的 labels/
    weights_dir = weights_dir or WEIGHTS_DIR
    torch.manual_seed(seed)
    train_x, train_y = to_tensors(load_samples(fold_paths(fold, "train", base_dir) + list(extra or [])), config["size"])
    val_samples = load_samples(fold_paths(fold, "val", base_dir))

    model = build_model(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=UNET_ARGS["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    best_path = weights_dir / f"{name}_fold{fold}.pth"

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
            x, y = augment(train_x[idx].to(DEVICE), train_y[idx].to(DEVICE), config["aug"] == "strong")
            if config.get("jpeg"):
                x = jpeg_augment(x, *config["jpeg"])
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
        if dice > best_dice and save_best({"epoch": epoch, "model_state_dict": model.state_dict(), "mean_dice": dice}, best_path):
            best_dice = dice
        print(f"[{name} fold{fold}] epoch {epoch}/{epochs} loss {sum(losses) / len(losses):.4f} val dice {dice:.4f} (best {best_dice:.4f})")

    # 訓練完整跑完且有成功存下權重才留下標記,否則下次會重新訓練
    if best_dice >= 0:
        best_path.with_suffix(".done").touch()
    del model


def evaluate(name, config, folds, dataset, test_samples):
    rows = []
    for fold in folds:
        model = build_model({**config, "pretrained": False}).to(DEVICE)
        model.load_state_dict(torch.load(WEIGHTS_DIR / f"{name}_fold{fold}.pth", map_location=DEVICE)["model_state_dict"])
        model.eval()
        samples = load_samples(fold_paths(fold, "val")) if dataset == "cv" else test_samples
        for path, image, gt in samples:
            masks = predict_masks(model, image, config["size"])
            for cls in NAMES.values():
                row = {"dataset": dataset, "fold": fold, "filename": path.stem, "variant": name, "class": cls}
                rows.append({**row, **pixel_metrics(masks[cls], gt[cls])})
        del model
    return rows


def reference_rows(path, variant, tag):
    # 先前已算好的結果(原本的 U-Net 與 YOLO),改用易懂的名稱後放進同一張表
    if not path.exists():
        print(f"[注意] 找不到 {path},表中不會列出 {tag}")
        return []
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["variant"] == variant]
    for r in rows:
        r["variant"] = tag
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(K_FOLDS)))
    parser.add_argument("--epochs", type=int, default=UNET_ARGS["epochs"])
    parser.add_argument("--images", default=IMAGES_DIR, help="獨立測試集的影像資料夾")
    parser.add_argument("--gt", default=GT_DIR, help="獨立測試集的標註資料夾")
    parser.add_argument("--eval-only", action="store_true", help="不訓練,只用已訓練好的折評分")
    parser.add_argument("--baseline", default=BASE_TAG, help=f"成對比較的基準,可填 {BASE_TAG}、{YOLO_VARIANT} 或任一設定名稱")
    args = parser.parse_args()

    # 輸出導向檔案(nohup)時預設會整塊緩衝,改成逐行寫出才能即時看到進度
    sys.stdout.reconfigure(line_buffering=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.eval_only:
        # 先把每種模型都建一次:預訓練權重下載失敗或設定有誤時,在開始訓練前就會發現
        for name in args.configs:
            build_model(CONFIGS[name])
        print(f"模型檢查完成,開始訓練: {args.configs}")
        # 先跑完一整折的所有設定再換下一折,中途的結果也能互相比較
        for fold in args.folds:
            for name in args.configs:
                if (WEIGHTS_DIR / f"{name}_fold{fold}.done").exists():
                    print(f"[已完成] {name} fold={fold}")
                    continue
                train(name, CONFIGS[name], fold, args.epochs)
                gc.collect()
                torch.cuda.empty_cache()

    test_gt = (ROOT / args.gt).resolve()
    test_samples = []
    for path in index_files((ROOT / args.images).resolve()).values():
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        test_samples.append((path, image, load_gt(path.stem, image.shape, test_gt, {})))

    for dataset in ("cv", "test"):
        rows = reference_rows(base_csv_path(dataset), "unet", BASE_TAG)
        if not rows:
            raise SystemExit(f"缺少原本 U-Net 的結果,請先執行 eval_kfold_variants.py --dataset {dataset}")
        rows += reference_rows(yolo_csv_path(dataset), YOLO_VARIANT, YOLO_VARIANT)
        for name in CONFIGS:
            folds = [f for f in range(K_FOLDS) if (WEIGHTS_DIR / f"{name}_fold{f}.done").exists()]
            if folds:
                rows += evaluate(name, CONFIGS[name], folds, dataset, test_samples)
                print(f"[{dataset}] {name}: 已完成的折 {folds}")

        merged_csv = OUT_DIR / f"compare_{dataset}.csv"
        with open(merged_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        summarize(dataset, merged_csv, args.baseline)
    print("\n[注意] 尚未跑完五折的設定只包含已完成的折,和其他設定的比較基礎不同,請以五折都完成後的結果為準")


if __name__ == "__main__":
    main()
