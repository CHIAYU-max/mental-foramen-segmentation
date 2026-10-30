import argparse
import itertools
import statistics
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stage2.compare_yolo_unet import GT_DIR, IMAGES_DIR, NAMES, ROOT, index_files, load_gt, pixel_metrics

WEIGHTS = "best_unet(unsharp_mask).pth"
# 輸出的遮罩為類別索引圖(0 背景, 1 mf, 2 ian),檔名與影像相同,可直接給 compare_yolo_unet.py --unet 使用
OUT_DIR = "runs/unet/test_um_masks"

# 訓練時的輸入尺寸、正規化與 skip connection 串接順序已無紀錄,以下組合全部試一遍,取與 GT 最接近的
SIZES = [256, 384, 512, 640, "native"]
NORMS = {
    "div255": lambda x: x,
    "mean0.5_std0.5": lambda x: (x - 0.5) / 0.5,
    "imagenet_gray": lambda x: (x - 0.449) / 0.226,
    "zscore": lambda x: (x - x.mean()) / (x.std() + 1e-6),
}
# up_skip: cat([上採樣, skip]);skip_up: cat([skip, 上採樣])
CATS = ["up_skip", "skip_up"]
# unet_predict.py 在縮放前先做雙邊濾波(d=9, sigmaColor=75, sigmaSpace=75),但不確定這份權重訓練時是否有用
FILTERS = {
    "none": lambda image: image,
    "bilateral": lambda image: cv2.bilateralFilter(image, 9, 75, 75),
}
# 已由 unet_predict.py 確認的設定,預設只試這些;其餘選項可用參數指定
DEFAULT_NORMS = ["div255"]
DEFAULT_CATS = ["up_skip"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DoubleConv(nn.Module):
    def __init__(self, c_in, c_mid, c_out, bias):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_mid, 3, padding=1, bias=bias),
            nn.BatchNorm2d(c_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_out, 3, padding=1, bias=bias),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    # 沒有原始的模型定義,結構直接由權重各層的名稱與形狀推回
    def __init__(self, state_dict):
        super().__init__()
        blocks, ups, outs = {}, {}, {}
        for key, value in state_dict.items():
            if key.endswith(".block.0.weight"):
                blocks[key[: -len(".block.0.weight")]] = value
            elif key.endswith(".weight") and value.ndim == 4 and ".block." not in key:
                (outs if value.shape[-1] == 1 else ups)[key[: -len(".weight")]] = value

        # 通道數變多的是 encoder(最深的一層是 bottleneck),變少的是 decoder
        def shape(name):
            c_out = state_dict[f"{name}.block.3.weight"].shape[0]
            return blocks[name].shape[1], blocks[name].shape[0], c_out

        enc_names = sorted((n for n in blocks if shape(n)[2] > shape(n)[0]), key=lambda n: shape(n)[2])
        dec_names = sorted((n for n in blocks if shape(n)[2] < shape(n)[0]), key=lambda n: -shape(n)[2])
        up_names = sorted(ups, key=lambda n: -ups[n].shape[1])
        if len(outs) != 1 or not len(dec_names) == len(up_names) == len(enc_names) - 1:
            raise SystemExit(
                "無法由權重推回 U-Net 結構\n"
                f"  encoder: {enc_names}\n  decoder: {dec_names}\n  upconv: {up_names}\n  output: {list(outs)}"
            )
        out_name = next(iter(outs))

        def double_conv(name):
            return DoubleConv(*shape(name), bias=f"{name}.block.0.bias" in state_dict)

        self.encs = nn.ModuleList(double_conv(n) for n in enc_names)
        self.decs = nn.ModuleList(double_conv(n) for n in dec_names)
        self.ups = nn.ModuleList(
            nn.ConvTranspose2d(ups[n].shape[0], ups[n].shape[1], ups[n].shape[-1], stride=2, bias=f"{n}.bias" in state_dict)
            for n in up_names
        )
        self.out = nn.Conv2d(outs[out_name].shape[1], outs[out_name].shape[0], 1, bias=f"{out_name}.bias" in state_dict)
        self.in_channels = shape(enc_names[0])[0]
        self.stride = 2 ** len(up_names)
        self.cat = CATS[0]

        # 把原本的層名稱對應到這裡的名稱再載入,名稱對不上的會直接報錯
        prefixes = {n: f"encs.{i}" for i, n in enumerate(enc_names)}
        prefixes.update({n: f"decs.{i}" for i, n in enumerate(dec_names)})
        prefixes.update({n: f"ups.{i}" for i, n in enumerate(up_names)})
        prefixes[out_name] = "out"
        remapped = {}
        for key, value in state_dict.items():
            name = max((n for n in prefixes if key.startswith(n + ".")), key=len, default=None)
            if name is None:
                raise SystemExit(f"權重中有無法對應的層: {key}")
            remapped[prefixes[name] + key[len(name) :]] = value
        self.load_state_dict(remapped)
        print(f"U-Net 結構: encoder {enc_names}, decoder {dec_names}, upconv {up_names}, output {out_name}")

    def forward(self, x):
        skips = []
        for enc in self.encs[:-1]:
            x = enc(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.encs[-1](x)
        for up, dec, skip in zip(self.ups, self.decs, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([x, skip] if self.cat == "up_skip" else [skip, x], dim=1))
        return self.out(x)


@torch.inference_mode()
def predict(model, image, size, norm):
    # 回傳原圖解析度的類別索引圖
    h, w = image.shape[:2]
    x = image.astype(np.float32) / 255
    if size != "native":
        x = cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
    x = NORMS[norm](x)
    x = torch.from_numpy(x).to(DEVICE)
    x = x[None, None] if x.ndim == 2 else x.permute(2, 0, 1)[None]
    # 原尺寸輸入時補邊到下採樣倍率的整數倍,推論後再裁掉
    in_h, in_w = x.shape[-2:]
    x = F.pad(x, (0, -in_w % model.stride, 0, -in_h % model.stride), mode="replicate")
    logits = model(x)[..., :in_h, :in_w]
    logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
    return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)


def parse_size(value):
    return value if value == "native" else int(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=WEIGHTS)
    parser.add_argument("--images", default=IMAGES_DIR)
    parser.add_argument("--gt", default=GT_DIR)
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--size", type=parse_size, nargs="+", default=SIZES, help="只試指定的輸入尺寸(整數或 native)")
    parser.add_argument("--norm", nargs="+", default=DEFAULT_NORMS, choices=list(NORMS), help="只試指定的正規化方式")
    parser.add_argument("--cat", nargs="+", default=DEFAULT_CATS, choices=CATS, help="只試指定的 skip 串接順序")
    parser.add_argument("--filter", nargs="+", default=list(FILTERS), choices=list(FILTERS), help="只試指定的濾波方式")
    args = parser.parse_args()

    images_dir = (ROOT / args.images).resolve()
    gt_dir = (ROOT / args.gt).resolve()
    out_dir = (ROOT / args.out).resolve()

    checkpoint = torch.load(ROOT / args.weights, map_location="cpu", weights_only=False)
    stored = {k: v for k, v in checkpoint.items() if not k.endswith("state_dict")}
    model = UNet(checkpoint["model_state_dict"]).to(DEVICE).eval()
    if model.out.out_channels != len(NAMES) + 1:
        raise SystemExit(f"模型輸出 {model.out.out_channels} 個通道,與 背景 + {list(NAMES.values())} 不符")

    files = index_files(images_dir)
    if not files:
        raise SystemExit(f"{images_dir}: 找不到任何影像")
    gt_files = index_files(gt_dir)
    flag = cv2.IMREAD_GRAYSCALE if model.in_channels == 1 else cv2.IMREAD_COLOR
    images, gts = {}, {}
    for stem, path in files.items():
        images[stem] = cv2.imread(str(path), flag)
        if images[stem] is None:
            raise SystemExit(f"無法讀取影像: {path}")
        gts[stem] = load_gt(stem, images[stem].shape[:2], gt_dir, gt_files)

    # 每種組合都對全部測試影像推論,以逐張 Dice 的平均排序
    results = []
    filtered = {name: {stem: FILTERS[name](image) for stem, image in images.items()} for name in args.filter}
    variants = list(itertools.product(args.filter, args.cat, args.size, args.norm))
    for i, (flt, cat, size, norm) in enumerate(variants, start=1):
        model.cat = cat
        dice = {cls: [] for cls in NAMES.values()}
        for stem, image in filtered[flt].items():
            pred = predict(model, image, size, norm)
            for c, cls in NAMES.items():
                dice[cls].append(pixel_metrics(pred == c + 1, gts[stem][cls])["dice"])
        item = {"filter": flt, "cat": cat, "size": size, "norm": norm}
        item.update({cls: statistics.mean(values) for cls, values in dice.items()})
        item["mean"] = statistics.mean(item[cls] for cls in NAMES.values())
        results.append(item)
        print(f"[{i}/{len(variants)}] filter={flt} cat={cat} size={size} norm={norm}: mean dice {item['mean']:.4f}")
    results.sort(key=lambda x: x["mean"], reverse=True)

    print(f"\n各組合在 {len(images)} 張測試影像上的逐張平均 Dice(由高到低)")
    header = f"{'filter':<11}{'cat':<10}{'size':>8}{'norm':>18}"
    print(header + "".join(f"{cls + '_dice':>12}" for cls in NAMES.values()) + f"{'mean_dice':>12}")
    for item in results:
        cells = "".join(f"{item[cls]:>12.4f}" for cls in NAMES.values())
        print(f"{item['filter']:<11}{item['cat']:<10}{str(item['size']):>8}{item['norm']:>18}{cells}{item['mean']:>12.4f}")
    print(f"\n權重檔內記錄的訓練時驗證結果(供對照): {stored}")

    best = results[0]
    model.cat = best["cat"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem, image in filtered[best["filter"]].items():
        cv2.imwrite(str(out_dir / f"{stem}.png"), predict(model, image, best["size"], best["norm"]))
    print(
        f"\n已用最佳組合 filter={best['filter']} cat={best['cat']} size={best['size']} norm={best['norm']} "
        f"輸出 {len(images)} 張遮罩: {out_dir}"
    )


if __name__ == "__main__":
    main()
