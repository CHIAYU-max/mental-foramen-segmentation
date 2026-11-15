"""2026-11-05 針對頦孔定位(最受畫質影響的階段)試推論端的方法,不重新訓練(GPU,輸出 compare_results/quality_errors/methods/)

背景(quality_errors.py): 沒命中與 ROI 對比低、整體偏亮、頦孔 CNR 低有關;第二批命中 74.7%(第一批 97.4%);
沒命中的種類 CV: 錯到別處 far 52%、偏一點 near 30%、沒有預測 empty 15%。
模型: R34/512 主結果種子 0 的五折權重(其他種子的權重已刪)。CV = 每張用沒看過它的那一折;測試 = 五折多數決。
方法(門檻與參數事先固定;只用 CV 挑,測試只確認被選中的方法):
  V0 control  原流程(argmax → lcc),應與已存遮罩相同
  V1 histmatch 推論前把影像直方圖對齊到該折訓練影像的平均 CDF(不用標註)
  V2 zscore   推論前把影像線性調成該折訓練影像的平均 / 標準差
  V3 gammaTTA gamma 0.67 / 1 / 1.5 三種輸入的 softmax 平均
  V4 fallback 原流程沒有頦孔時,改取 頦孔機率 ≥ 0.3 的最大區塊(只影響 empty)
  V5 probmass 連通區塊改取「頦孔機率總和」最大者(取代面積最大)
  V6 nearian  連通區塊改取離預測下齒槽神經管最近者(沒有 ian 預測時用 lcc)
另算兩個上限(oracle,用 GT,只說明空間): 原始 argmax 的某個區塊命中;頦孔機率 ≥ 0.3 的某個區塊命中。
統計: 與 V0 配對,以 folds.csv 的組(病人 ∪ 全景片)bootstrap 2000 次;另報第二批、可辨識度分層。
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from stage2.compare_yolo_unet import GT_DIR, IMAGES_DIR, NAMES, ROOT, index_files, load_gt
from stage2.kfold_pano import EXCLUDE_TEST, PROJECT
from stage2.kfold_unet_variants import CONFIGS, build_model, fold_paths
from stage2.kfold_yolo_unet import K_FOLDS
from stage2.mf_postproc import largest
from analysis.paper_metrics import read_masks
from analysis.pano_visibility import VISIBILITY_CSV
from stage2.predict_unet import DEVICE

MASKS = ROOT / "compare_results" / PROJECT
OUT = ROOT / "compare_results" / "quality_errors" / "methods"
CFG = CONFIGS["r34_imagenet_aug"]
VARIANTS = ["V0 control", "V1 histmatch", "V2 zscore", "V3 gammaTTA", "V4 fallback", "V5 probmass", "V6 nearian"]
LOW = 0.3


def load_model(fold):
    m = build_model({**CFG, "pretrained": False}).to(DEVICE)
    m.load_state_dict(torch.load(MASKS / "weights" / f"r34_imagenet_aug_fold{fold}.pth", map_location=DEVICE)["model_state_dict"])
    return m.eval()


@torch.inference_mode()
def probs(model, img):
    h, w = img.shape
    x = cv2.resize(img, (CFG["size"][1], CFG["size"][0]), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    logits = model(torch.from_numpy(x)[None, None].to(DEVICE))
    return F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False).softmax(1)[0].cpu().numpy()


def train_stats(fold):
    hist = np.zeros(256)
    means, stds = [], []
    for p in fold_paths(fold, "train", MASKS):
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        h = np.bincount(im.ravel(), minlength=256).astype(float)
        hist += h / h.sum()
        means.append(im.mean()); stds.append(im.std())
    cdf = np.cumsum(hist / hist.sum())
    return cdf, float(np.mean(means)), float(np.mean(stds))


def histmatch(img, ref_cdf):
    h = np.bincount(img.ravel(), minlength=256).astype(float)
    cdf = np.cumsum(h / h.sum())
    lut = np.clip(np.searchsorted(ref_cdf, cdf), 0, 255).astype(np.uint8)
    return lut[img]


def zscore(img, mu, sd):
    f = (img.astype(np.float32) - img.mean()) / (img.std() + 1e-6) * sd + mu
    return np.clip(f, 0, 255).astype(np.uint8)


def gamma(img, g):
    return np.clip(255 * (img.astype(np.float32) / 255) ** g, 0, 255).astype(np.uint8)


def components(mask):
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return [lab == i for i in range(1, n)]


def select(p, mode):
    # p: softmax (3, h, w);回傳頦孔遮罩
    arg = p.argmax(0)
    mf, ian = arg == 1, arg == 2
    if mode == "fallback" and not mf.any():
        return largest(p[1] >= LOW)
    if mode in ("probmass", "nearian") and mf.any():
        comps = components(mf)
        if mode == "probmass":
            return max(comps, key=lambda c: p[1][c].sum())
        if ian.any():
            dist = cv2.distanceTransform((~ian).astype(np.uint8), cv2.DIST_L2, 3)
            return min(comps, key=lambda c: dist[c].min())
    return largest(mf)


def score(pred, gt):
    r = (gt.sum() / np.pi) ** 0.5
    inter = (pred & gt).sum()
    dice = 2 * inter / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) else 1.0
    if not pred.any():
        return 0.0, float(dice), "empty"
    e = np.linalg.norm(np.argwhere(pred).mean(0) - np.argwhere(gt).mean(0)) / r
    return float(e <= 1), float(dice), "hit" if e <= 1 else ("near" if e <= 2 else "far")


def any_hit(mask, gt):
    r = (gt.sum() / np.pi) ** 0.5
    g = np.argwhere(gt).mean(0)
    return float(any(np.linalg.norm(np.argwhere(c).mean(0) - g) <= r for c in components(mask)))


def variant_masks(model, img, stats):
    # 回傳各方法「取區塊前」的頦孔遮罩: V0–V3 為 argmax 結果(之後 lcc),V4–V6 為已選好的區塊
    # CV: largest(遮罩);測試: largest(五折多數決),與主流程「先投票再 lcc」相同
    cdf, mu, sd = stats
    p0 = probs(model, img)
    out = {"V0 control": p0.argmax(0) == 1,
           "V1 histmatch": probs(model, histmatch(img, cdf)).argmax(0) == 1,
           "V2 zscore": probs(model, zscore(img, mu, sd)).argmax(0) == 1,
           "V3 gammaTTA": np.mean([p0] + [probs(model, gamma(img, g)) for g in (0.67, 1.5)], axis=0).argmax(0) == 1,
           "V4 fallback": select(p0, "fallback"), "V5 probmass": select(p0, "probmass"), "V6 nearian": select(p0, "nearian")}
    return out, p0


def group_boot(diffs, groups, n=2000, seed=0):
    g = defaultdict(list)
    for d, k in zip(diffs, groups):
        g[k].append(d)
    keys = list(g)
    sums = np.array([sum(g[k]) for k in keys]); cnts = np.array([len(g[k]) for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), (n, len(keys)))
    st = sums[idx].sum(1) / cnts[idx].sum(1)
    lo, hi = np.percentile(st, [2.5, 97.5])
    return float(lo), float(hi), float(min(1.0, 2 * min((st <= 0).mean(), (st >= 0).mean())))


def main():
    sys.stdout.reconfigure(line_buffering=True)
    OUT.mkdir(parents=True, exist_ok=True)
    folds = {r["stem"]: r for r in csv.DictReader(open(MASKS / "folds.csv", newline=""))}
    visible = {r["stem"]: r["mf_visible"] for r in csv.DictReader(open(VISIBILITY_CSV, newline=""))}
    rows, control_diff = [], 0
    models = {f: load_model(f) for f in range(K_FOLDS)}
    stats = {f: train_stats(f) for f in range(K_FOLDS)}
    for fold in range(K_FOLDS):
        for path in fold_paths(fold, "val", MASKS):
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            gt = load_gt(path.stem, img.shape, path.parent.parent / "labels", {})["mf"]
            if not gt.any():
                continue
            masks, p0 = variant_masks(models[fold], img, stats[fold])
            saved = largest(read_masks(MASKS, "cv", "r34_imagenet_aug", fold, path.stem)["mf"])
            control_diff += int((saved != largest(masks["V0 control"])).sum() > 0)
            row = {"setting": "cv", "stem": path.stem, "group": folds[path.stem]["group"], "batch": path.stem.split("_")[0],
                   "visible": visible.get(path.stem, ""),
                   "oracle_argmax": any_hit(p0.argmax(0) == 1, gt), "oracle_p03": any_hit(p0[1] >= LOW, gt)}
            for v, m in masks.items():
                row[v + "|hit"], row[v + "|dice"], row[v + "|type"] = score(largest(m), gt)
            rows.append(row)
        print("cv fold", fold, "done", flush=True)
    test_gt = (ROOT / GT_DIR).resolve()
    for stem, path in sorted(index_files((ROOT / IMAGES_DIR).resolve()).items()):
        if stem in EXCLUDE_TEST:
            continue
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        gt = load_gt(stem, img.shape, test_gt, {})["mf"]
        if not gt.any():
            continue
        per = [variant_masks(models[f], img, stats[f])[0] for f in range(K_FOLDS)]
        row = {"setting": "test", "stem": stem, "group": stem.split("_")[0], "batch": "test", "visible": visible.get(stem, ""),
               "oracle_argmax": "", "oracle_p03": ""}
        for v in VARIANTS:
            vote = largest(np.sum([m[v] for m in per], axis=0) > K_FOLDS / 2)
            row[v + "|hit"], row[v + "|dice"], row[v + "|type"] = score(vote, gt)
        rows.append(row)
    print("test done; CV 對照組與已存遮罩不同的 ROI:", control_diff, flush=True)
    with open(OUT / "per_roi.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lines = ["# 頦孔定位: 推論端的方法(quality_methods.py,種子 0)\n",
             f"- 對照組檢查: CV 對照組與已存遮罩(lcc)不同的 ROI {control_diff} 個(應為 0)"]
    cv = [r for r in rows if r["setting"] == "cv"]
    lines.append(f"- 上限(用 GT): CV 原始 argmax 有某個區塊命中 {100 * np.mean([r['oracle_argmax'] for r in cv]):.1f}%;"
                 f"頦孔機率 ≥ 0.3 有某個區塊命中 {100 * np.mean([r['oracle_p03'] for r in cv]):.1f}%")
    for setting in ("cv", "test"):
        sub = [r for r in rows if r["setting"] == setting]
        lines.append(f"\n## {setting}({len(sub)} 個)\n")
        lines.append("| 方法 | 命中 | 差(− V0)[95% CI] | p | mf Dice | 差 | 第二批命中 | 無法辨識命中 | 沒命中種類 empty / near / far |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for v in VARIANTS:
            h = np.array([r[v + "|hit"] for r in sub]); h0 = np.array([r["V0 control|hit"] for r in sub])
            d = np.array([r[v + "|dice"] for r in sub]); d0 = np.array([r["V0 control|dice"] for r in sub])
            if v == "V0 control":
                ci = "— | —"
            else:
                lo, hi, p = group_boot(h - h0, [r["group"] for r in sub])
                ci = f"{100 * (h - h0).mean():+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}] | {p:.3f}"
            sec = [r for r in sub if r["batch"] == "second"]
            inv = [r for r in sub if r["visible"] == "0"]
            t = [r[v + "|type"] for r in sub]
            lines.append(f"| {v} | {100 * h.mean():.2f}% | {ci} | {d.mean():.4f} | {d.mean() - d0.mean():+.4f} | "
                         + (f"{100 * np.mean([r[v + '|hit'] for r in sec]):.1f}%" if sec else "—") + " | "
                         + (f"{100 * np.mean([r[v + '|hit'] for r in inv]):.1f}%" if inv else "—") + " | "
                         + f"{t.count('empty')} / {t.count('near')} / {t.count('far')} |")
    text = "\n".join(lines)
    (OUT / "summary.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
