import argparse
import csv
import time
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stage2.compare_yolo_unet import NAMES
from deployment.int8_quant import read_split, unet_input
from stage2.kfold_unet_variants import augment, fold_paths, to_tensors, val_dice
from stage2.kfold_yolo_unet import UNET_ARGS, dice_loss, load_samples
from stage2.predict_unet import DEVICE
from hardware.qat_fx import OUT, PANO, SIZE
from hardware.qat_int4 import load_fp32

warnings.filterwarnings("ignore")
# 2:4 結構化稀疏的精度(2026-11-10,伺服器 GPU,使用者要求;見 EXPERIMENTS.md「硬體: 結構化稀疏與 4-bit 的評估與規劃」)。
#   accel_lowbit.py 顯示 2:4 是分析模型裡延遲與能耗最大的槓桿(樂觀上限),所以照事先定的條件做精度。FP32,不量化。
#   剪枝(事先定): 所有 in_channels 是 4 的倍數的 Conv2d(只跳過 3 通道的 stem),每個 (輸出通道, kh, kw) 沿輸入通道每 4 個
#     依絕對值留 2 個(與 accel_lowbit.py 的 K 方向一致);從 R34/512 主結果權重開始,一次剪完。
#   微調: 預算同 qat_fx.py(--epochs 15、AdamW --lr 3e-5 cosine、batch / weight decay 同主結果、strong 增強),
#     每步之後重新套遮罩;取驗證 Dice 最佳的 epoch(含 epoch 0 = 剪完不微調)。BN 正常更新。
#   變體: sp24_prune(epoch 0)、sp24_ft;遮罩存到 compare_results/int8_pano/masks/{cv,test}/<變體>/fold<k>/,
#   和 ov_fp32(= 主結果遮罩)用 slim_eval.py 配對比較。


def prune_24(model, dense=()):
    masks, kept, total = {}, 0.0, 0.0
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            w = m.weight.data
            total += w.numel()
            if m.in_channels % 4 or m.groups != 1 or name in dense:
                kept += w.numel()
                continue
            o, i, kh, kw = w.shape
            g = w.permute(0, 2, 3, 1).reshape(-1, 4)
            idx = g.abs().topk(2, dim=1).indices
            mask = torch.zeros_like(g).scatter_(1, idx, 1.0).reshape(o, kh, kw, i).permute(0, 3, 1, 2).contiguous()
            m.weight.data.mul_(mask)
            masks[name] = mask
            kept += mask.sum().item()
    return masks, kept / total


def apply_masks(model, masks):
    mods = dict(model.named_modules())
    with torch.no_grad():
        for name, mask in masks.items():
            mods[name].weight.mul_(mask)


def save_masks(model, variant, fold, sets):
    model.eval()
    for dataset, paths in sets.items():
        folder = OUT / "masks" / dataset / variant / f"fold{fold}"
        folder.mkdir(parents=True, exist_ok=True)
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            h, w = image.shape
            with torch.inference_mode():
                logits = model(torch.from_numpy(unet_input(image, SIZE)).to(DEVICE)).float()
            pred = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False).argmax(1)[0].cpu().numpy()
            code = np.zeros((h, w), dtype=np.uint8)
            for c in NAMES:
                code[pred == c + 1] = 1 << c
            cv2.imwrite(str(folder / f"{path.stem}.png"), code)


def one(seed, fold, args):
    suffix = ("_rt" if args.fp32_project else "") + (f"_s{seed}" if seed else "")
    # --dense 有給時變體前綴改成 sp24d(2026-11-10 第二輪: 輸出頭最後一層 head.1 保留 dense,見 EXPERIMENTS.md)
    pre = "sp24d" if args.dense else "sp24"
    done = OUT / "models" / f"{pre}{suffix}_fold{fold}.done"
    if done.exists():
        return
    start = time.time()
    torch.manual_seed(seed)
    model = load_fp32(seed, fold, args.fp32_project).to(DEVICE)
    masks, density = prune_24(model, set(args.dense))
    print(f"[{pre} s{seed} fold{fold}] 剪枝後保留權重 {density:.1%},剪了 {len(masks)} 層,dense {args.dense}", flush=True)
    prune_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    train_x, train_y = to_tensors(load_samples(fold_paths(fold, "train", PANO)), SIZE)
    val_samples = load_samples(fold_paths(fold, "val", PANO))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=UNET_ARGS["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    best_dice, best_state = val_dice(model, val_samples, SIZE), prune_state
    log = [(0, best_dice)]
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_x))
        for i in range(0, len(order), UNET_ARGS["batch"]):
            idx = order[i : i + UNET_ARGS["batch"]]
            if len(idx) < 2:
                continue
            x, y = augment(train_x[idx].to(DEVICE), train_y[idx].to(DEVICE), True)
            logits = model(x)
            loss = F.cross_entropy(logits, y) + dice_loss(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            apply_masks(model, masks)
        scheduler.step()
        dice = val_dice(model, val_samples, SIZE)
        log.append((epoch, dice))
        if dice > best_dice:
            best_dice, best_state = dice, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    print(f"[{pre} s{seed} fold{fold}] val dice 剪枝 {log[0][1]:.4f} → 最佳 {best_dice:.4f};訓練 {time.time() - start:.0f} s",
          flush=True)

    sets = {"cv": read_split(f"fold{fold}_val")[:args.limit], "test": read_split("test")[:args.limit]}
    torch.save(best_state, OUT / "models" / f"{pre}_ft{suffix}_fold{fold}.pth")
    model.load_state_dict(prune_state)
    save_masks(model, f"{pre}_prune{suffix}", fold, sets)
    model.load_state_dict(best_state)
    save_masks(model, f"{pre}_ft{suffix}", fold, sets)
    with open(OUT / f"{pre}_val_log.csv", "a", newline="") as f:
        csv.writer(f).writerows((seed, fold, e, round(d, 5)) for e, d in log)
    done.touch()
    print(f"[{pre} s{seed} fold{fold}] 完成,累計 {time.time() - start:.0f} s", flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fp32-project", default=None, help="FP32 起點改讀 compare_results/<這個>/weights/(重訓的種子);輸出名稱加 _rt")
    parser.add_argument("--dense", nargs="*", default=[], help="不剪的 Conv2d 名稱(例如 head.1)")
    args = parser.parse_args()
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for fold in args.folds:
            one(seed, fold, args)


if __name__ == "__main__":
    main()
