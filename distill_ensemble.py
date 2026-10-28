import argparse
import csv
import math
import statistics
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from compare_preprocessing import collect_images
from compare_yolo_unet import GT_DIR, IMAGES_DIR, NAMES, ROOT, index_files, load_gt
from kfold_pano import EXCLUDE_TEST, PROJECT
from kfold_patient import DATASET
from kfold_unet_variants import CONFIGS, build_model, predict_masks, to_tensors
from kfold_yolo_unet import K_FOLDS, UNET_ARGS, dice_loss, load_samples
from paper_metrics import image_metrics
from predict_unet import DEVICE

# 2026-10-27 把五折多數決蒸餾成一個 U-Net(SOP 用,使用者同意 GPU 空閒時排實驗):
#   老師 = kfold_pano_mf_559 的 r34_imagenet_aug 種子 0 五折模型,對 559 張訓練 ROI 的 softmax 平均(不做增強)。
#   學生 = 同一個 U-Net 設定,用全部 559 張訓練,固定 100 epoch、取最後一個 epoch(沒有驗證集可挑)。
#   兩個條件: full(只用 GT 的 CE + Dice,對照組)與 kd(再加上對老師 soft label 的交叉熵,權重 --alpha)。
#   評估: 獨立測試集 137 張的 mf 命中率與 Dice,和五折多數決(5 倍運算)、單一折模型平均比較。
#   輸出: compare_results/kfold_pano_mf_559/distill/summary.csv;權重在同目錄 weights/。
#   2026-10-27 加 --student: 學生可以用較小的設定(例如 r18_imagenet_aug_384);老師仍在 512 上推論,
#   soft label 以雙線性內插縮到學生的輸入大小。
CONFIG = "r34_imagenet_aug"
OUT_DIR = ROOT / "compare_results" / PROJECT
DISTILL_DIR = OUT_DIR / "distill"


@torch.inference_mode()
def teacher_probs(x):
    # 五折模型的 softmax 平均,存成 float16 放在 CPU(559 × 3 × 512 × 512 約 0.9 GB)
    total = torch.zeros(len(x), len(NAMES) + 1, *x.shape[-2:])
    for fold in range(K_FOLDS):
        model = build_model({**CONFIGS[CONFIG], "pretrained": False}).to(DEVICE)
        model.load_state_dict(torch.load(OUT_DIR / "weights" / f"{CONFIG}_fold{fold}.pth", map_location=DEVICE)["model_state_dict"])
        model.eval()
        for i in range(0, len(x), 16):
            with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
                total[i : i + 16] += F.softmax(model(x[i : i + 16].to(DEVICE)).float(), 1).cpu()
        del model
    return (total / K_FOLDS).half()


def augment_with_soft(x, y, soft):
    # 與 kfold_unet_variants.augment(strong)相同的隨機變換,同時套用到 soft label(雙線性內插)
    n = len(x)
    flip = torch.rand(n, device=x.device) < 0.5
    x[flip], y[flip], soft[flip] = x[flip].flip(-1), y[flip].flip(-1), soft[flip].flip(-1)
    angle = (torch.rand(n, device=x.device) - 0.5) * 2 * math.radians(15)
    scale = 0.8 + 0.4 * torch.rand(n, device=x.device)
    shift = (torch.rand(n, 2, device=x.device) - 0.5) * 0.4
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([torch.stack([cos, -sin, shift[:, 0]], dim=1), torch.stack([sin, cos, shift[:, 1]], dim=1)], dim=1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    y = F.grid_sample(y[:, None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)[:, 0].long()
    soft = F.grid_sample(soft, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    # 補零的區域與 y 一樣當成背景
    soft[:, 0] += (1 - soft.sum(1)).clamp(min=0)
    contrast = 1 + (torch.rand(n, 1, 1, 1, device=x.device) - 0.5) * 0.4
    brightness = (torch.rand(n, 1, 1, 1, device=x.device) - 0.5) * 0.2
    return ((x - 0.5) * contrast + 0.5 + brightness).clamp(0, 1), y, soft


def train_student(name, student, x, y, soft, alpha, epochs, seed):
    path = DISTILL_DIR / "weights" / f"{name}.pth"
    if path.exists():
        print(f"[已完成] {name}")
        return path
    config = CONFIGS[student]
    torch.manual_seed(seed)
    model = build_model(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=UNET_ARGS["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x))
        losses = []
        for i in range(0, len(order), UNET_ARGS["batch"]):
            idx = order[i : i + UNET_ARGS["batch"]]
            if len(idx) < 2:
                continue
            xb, yb, sb = augment_with_soft(x[idx].to(DEVICE), y[idx].to(DEVICE), soft[idx].to(DEVICE).float())
            with torch.autocast(DEVICE, enabled=use_amp):
                logits = model(xb)
                loss = F.cross_entropy(logits, yb) + dice_loss(logits, yb)
                if alpha:
                    loss = loss + alpha * -(sb * F.log_softmax(logits.float(), 1)).sum(1).mean()
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        scheduler.step()
        print(f"[{name}] epoch {epoch}/{epochs} loss {sum(losses) / len(losses):.4f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epochs, "model_state_dict": model.state_dict()}, path)
    del model
    return path


@torch.inference_mode()
def evaluate(path, student, test):
    model = build_model({**CONFIGS[student], "pretrained": False}).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE)["model_state_dict"])
    model.eval()
    rows = [image_metrics(predict_masks(model, image, CONFIGS[student]["size"]), gt) for image, gt in test]
    del model
    return {k: statistics.mean(r[k] for r in rows if r[k] == r[k]) for k in ("mf_hit_1R", "mf_dice", "ian_dice")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--alpha", type=float, default=1.0, help="soft label 交叉熵的權重")
    parser.add_argument("--epochs", type=int, default=UNET_ARGS["epochs"])
    parser.add_argument("--student", default=CONFIG, choices=list(CONFIGS), help="學生的 U-Net 設定")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    samples = load_samples(sorted(collect_images((ROOT / DATASET).resolve()).values()))
    x_teacher, _ = to_tensors(samples, CONFIGS[CONFIG]["size"])
    print(f"訓練 ROI: {len(x_teacher)} 張;計算老師(五折平均)的 soft label")
    soft = teacher_probs(x_teacher)
    del x_teacher
    x, y = to_tensors(samples, CONFIGS[args.student]["size"])
    if soft.shape[-2:] != x.shape[-2:]:
        soft = torch.cat([F.interpolate(soft[i : i + 32].float(), size=x.shape[-2:], mode="bilinear", align_corners=False).half()
                          for i in range(0, len(soft), 32)])

    test = []
    for stem, path in sorted(index_files((ROOT / IMAGES_DIR).resolve()).items()):
        if stem in EXCLUDE_TEST:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        test.append((image, load_gt(stem, image.shape, (ROOT / GT_DIR).resolve(), {})))
    print(f"測試 ROI: {len(test)} 張")

    results = []
    for seed in args.seeds:
        for cond, alpha in (("full", 0.0), ("kd", args.alpha)):
            name = f"{args.student}_{cond}559_a{args.alpha:g}_s{seed}" if cond == "kd" else f"{args.student}_{cond}559_s{seed}"
            path = train_student(name, args.student, x, y, soft, alpha, args.epochs, seed)
            results.append({"student": args.student, "condition": cond, "alpha": alpha, "seed": seed,
                             **{k: round(v, 4) for k, v in evaluate(path, args.student, test).items()}})
            print(results[-1])

    DISTILL_DIR.mkdir(parents=True, exist_ok=True)
    with open(DISTILL_DIR / f"summary_{args.student}_a{args.alpha:g}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    for cond in ("full", "kd"):
        hits = [r["mf_hit_1R"] for r in results if r["condition"] == cond]
        dice = [r["mf_dice"] for r in results if r["condition"] == cond]
        sd = statistics.stdev(hits) if len(hits) > 1 else 0.0
        print(f"{cond}: 測試命中 {statistics.mean(hits):.4f} ± {sd:.4f}, mf Dice {statistics.mean(dice):.4f} (n={len(hits)})")


if __name__ == "__main__":
    main()
