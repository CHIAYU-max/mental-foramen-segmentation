"""2026-11-05 哪一段最容易受影像畫質影響、錯在哪裡(CPU,只讀已存遮罩;輸出 compare_results/quality_errors/)

1. 第二階段(頦孔): CV 559 + 測試 137 個 ROI,R34/512 主結果 5 種子(CV = 該折模型、測試 = 五折多數決),lcc。
   - 每個 ROI 的影像品質(推論時拿得到): 平均亮度、對比(標準差、p5–p95)、清晰度(Laplacian 變異數)、雜訊(與 3×3 中值濾波的差的 MAD)、ROI 尺寸、批次。
   - 頦孔本身(要用 GT,只做分析): 等效半徑 R、頦孔對周圍 1R–2R 環的對比雜訊比 CNR、邊界梯度、到下齒槽神經管 GT 的距離 / R。
   - 錯誤種類(沒命中的每個種子): empty 沒有預測;near 1–2R;ian 預測落在下齒槽神經管上(GT ian 外擴 5 px);far 其他位置 > 2R。
   - 每個特徵對「多數種子沒命中」的 AUROC(> 0.5 = 數值越大越容易錯,< 0.5 = 越小越容易錯),以及依三分位的命中率。
2. 第一階段(牙齒 / FDI): 407 張全景片的品質(同上的全圖指標、解析度)對每側「兩種編號不一致」(fdi_flag)、參考線借用方向(key.csv axis_fallback)的 AUROC。
3. 牙位: 測試集正常牙,判錯 vs 頦孔是否命中、邊界距離(position_seeds/per_seed.csv)。
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from stage2.compare_yolo_unet import GT_DIR, IMAGES_DIR, ROOT, index_files, load_gt
from stage2.kfold_pano import EXCLUDE_TEST, PROJECT
from stage2.kfold_unet_variants import fold_paths
from stage2.kfold_yolo_unet import K_FOLDS
from stage2.mf_postproc import largest
from analysis.paper_metrics import read_masks
from analysis.pano_visibility import VISIBILITY_CSV
from deployment.pipeline_timing import PANO_DIR

MASKS = ROOT / "compare_results" / PROJECT
OUT = ROOT / "compare_results" / "quality_errors"
MP = ROOT / "compare_results" / "mf_position"
TAGS = ["r34_imagenet_aug"] + [f"r34_imagenet_aug_s{s}" for s in range(1, 5)]


def global_quality(img):
    f = img.astype(np.float32)
    med = cv2.medianBlur(img, 3).astype(np.float32)
    lo, hi = np.percentile(f, [5, 95])
    return {"mean": float(f.mean()), "std": float(f.std()), "range_p5_p95": float(hi - lo),
            "sharpness": float(cv2.Laplacian(f, cv2.CV_32F).var()),
            "noise": float(1.4826 * np.median(np.abs(f - med))), "height": img.shape[0], "width": img.shape[1]}


def local_quality(img, mf, ian):
    f = img.astype(np.float32)
    r = (mf.sum() / np.pi) ** 0.5
    k = lambda rad: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(rad) + 1, 2 * int(rad) + 1))
    m8 = mf.astype(np.uint8)
    ring = (cv2.dilate(m8, k(2 * r)) > 0) & ~(cv2.dilate(m8, k(r)) > 0) & ~ian
    inside = f[mf]
    out = f[ring] if ring.any() else f[~mf]
    gx, gy = cv2.Sobel(f, cv2.CV_32F, 1, 0), cv2.Sobel(f, cv2.CV_32F, 0, 1)
    edge = mf & ~(cv2.erode(m8, np.ones((3, 3), np.uint8)) > 0)
    c = np.argwhere(mf).mean(0)
    d_ian = float(np.min(np.linalg.norm(np.argwhere(ian) - c, axis=1))) / r if ian.any() else float("nan")
    return {"R": float(r), "cnr": float((out.mean() - inside.mean()) / (out.std() + 1e-6)),
            "edge_grad": float(np.hypot(gx, gy)[edge].mean()), "dist_ian_R": d_ian}


def classify_error(pred, gt_mf, gt_ian):
    # 回傳 (種類, 誤差 / R, 預測相對 GT 的位移 (dy, dx) / R)
    r = (gt_mf.sum() / np.pi) ** 0.5
    if not pred.any():
        return "empty", float("nan"), (float("nan"), float("nan"))
    c = np.argwhere(pred).mean(0)
    off = (c - np.argwhere(gt_mf).mean(0)) / r
    e = float(np.linalg.norm(off))
    off = (float(off[0]), float(off[1]))
    if e <= 1:
        return "hit", e, off
    if e <= 2:
        return "near", e, off
    ian5 = cv2.dilate(gt_ian.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    if ian5[int(round(c[0])), int(round(c[1]))]:
        return "ian", e, off
    return "far", e, off


def auroc(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    ok = ~np.isnan(s)
    s, y = s[ok], y[ok]
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), 0
    gt = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
    return float(gt), int(len(pos))


def stage2(lines):
    visible = {r["stem"]: int(r["mf_visible"]) for r in csv.DictReader(open(VISIBILITY_CSV, newline=""))}
    items = []
    for fold in range(K_FOLDS):
        for path in fold_paths(fold, "val", MASKS):
            items.append(("cv", fold, path.stem, path, path.parent.parent / "labels"))
    test_gt = (ROOT / GT_DIR).resolve()
    for stem, path in index_files((ROOT / IMAGES_DIR).resolve()).items():
        if stem not in EXCLUDE_TEST:
            items.append(("test", None, stem, path, test_gt))
    rows = []
    for setting, fold, stem, path, gt_dir in items:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        gt = load_gt(stem, img.shape, gt_dir, {})
        if not gt["mf"].any():
            continue
        row = {"setting": setting, "stem": stem, "batch": "test" if setting == "test" else stem.split("_")[0],
               "visible": visible.get(stem, ""), **global_quality(img), **local_quality(img, gt["mf"], gt["ian"])}
        types, offs = [], []
        for tag in TAGS:
            if setting == "cv":
                pred = read_masks(MASKS, "cv", tag, fold, stem)["mf"]
            else:
                preds = [read_masks(MASKS, "test", tag, f, stem)["mf"] for f in range(K_FOLDS)]
                pred = np.sum(preds, axis=0) > K_FOLDS / 2
            t, e, off = classify_error(largest(pred), gt["mf"], gt["ian"])
            types.append(t)
            offs.append(f"{off[0]:.2f},{off[1]:.2f}")
        row["types"] = "|".join(types)
        row["offsets"] = "|".join(offs)
        row["miss_rate"] = float(np.mean([t != "hit" for t in types]))
        rows.append(row)
    with open(OUT / "stage2_per_roi.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lines.append("## 1. 第二階段(頦孔): 錯誤種類與影像品質\n")
    for setting in ("cv", "test"):
        sub = [r for r in rows if r["setting"] == setting]
        types = Counter(t for r in sub for t in r["types"].split("|"))
        tot = sum(types.values())
        miss = tot - types["hit"]
        lines.append(f"- {setting}: {len(sub)} 個 ROI × 5 種子 = {tot};沒命中 {miss}({100 * miss / tot:.1f}%): "
                     + "、".join(f"{k} {types[k]}({100 * types[k] / max(miss, 1):.0f}%)" for k in ("empty", "near", "ian", "far")))
        for v in (1, 0):
            sv = [r for r in sub if r["visible"] == v]
            tv = Counter(t for r in sv for t in r["types"].split("|"))
            mv = sum(tv.values()) - tv["hit"]
            if sv:
                lines.append(f"  - 可辨識 = {v}: {len(sv)} 個,沒命中 {100 * mv / sum(tv.values()):.1f}%;"
                             + "、".join(f"{k} {tv[k]}" for k in ("empty", "near", "ian", "far")))
    feats = ["mean", "std", "range_p5_p95", "sharpness", "noise", "height", "width", "R", "cnr", "edge_grad", "dist_ian_R"]
    lines.append("\n| 特徵 | 推論時可得 | AUROC CV(多數種子沒命中) | AUROC 測試 | CV 命中率 低 / 中 / 高 三分位 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for ft in feats:
        cells = []
        for setting in ("cv", "test"):
            sub = [r for r in rows if r["setting"] == setting]
            a, n = auroc([r[ft] for r in sub], [r["miss_rate"] > 0.5 for r in sub])
            cells.append(f"{a:.2f}(錯 {n})")
        sub = [r for r in rows if r["setting"] == "cv" and not np.isnan(r[ft])]
        q = np.percentile([r[ft] for r in sub], [33.3, 66.7])
        terc = [np.mean([1 - r["miss_rate"] for r in sub if (r[ft] <= q[0] if i == 0 else (q[0] < r[ft] <= q[1] if i == 1 else r[ft] > q[1]))]) for i in range(3)]
        avail = "是" if ft in feats[:7] else "否(需 GT)"
        lines.append(f"| {ft} | {avail} | {cells[0]} | {cells[1]} | " + " / ".join(f"{100 * t:.1f}%" for t in terc) + " |")
    for b in sorted({r["batch"] for r in rows}):
        sb = [r for r in rows if r["batch"] == b]
        lines.append(f"- 批次 {b}: {len(sb)} 個,命中 {100 * np.mean([1 - r['miss_rate'] for r in sb]):.1f}%,ROI 尺寸中位數 {np.median([r['height'] for r in sb]):.0f}×{np.median([r['width'] for r in sb]):.0f}")
    lines.append("\n- 各批次的影像特徵(中位數): 批次 | 亮度 | 對比 std | p5–p95 | 清晰度 | 頦孔 R | 頦孔 CNR | 可辨識比例")
    for b in ("first", "second", "test"):
        sb = [r for r in rows if r["batch"] == b]
        md = lambda k: np.nanmedian([r[k] for r in sb])
        vis = [r["visible"] for r in sb if r["visible"] != ""]
        lines.append(f"  - {b}: {md('mean'):.0f} | {md('std'):.1f} | {md('range_p5_p95'):.0f} | {md('sharpness'):.0f} | {md('R'):.1f} | {md('cnr'):.2f} | {100 * np.mean(vis):.0f}%")
    # 錯到別處(far / ian)的位移方向: 影像座標 dy > 0 = 往下(遠離牙齒)
    offs = [tuple(map(float, o.split(","))) for r in rows if r["setting"] == "cv"
            for t, o in zip(r["types"].split("|"), r["offsets"].split("|")) if t in ("far", "ian")]
    if offs:
        dy = np.array([o[0] for o in offs]); dx = np.array([o[1] for o in offs]); e = np.hypot(dy, dx)
        lines.append(f"- CV 錯到別處 {len(offs)} 次: 距離中位數 {np.median(e):.1f} R(P25 {np.percentile(e, 25):.1f}、P75 {np.percentile(e, 75):.1f});"
                     f"往上(靠牙根){100 * np.mean(dy < -1):.0f}%、往下 {100 * np.mean(dy > 1):.0f}%、主要是水平(|dx| > |dy|){100 * np.mean(np.abs(dx) > np.abs(dy)):.0f}%")
    # 同一個 ROI 是不是每個種子都錯(資訊不足)還是只有部分種子錯(模型不穩)
    cvr = [r for r in rows if r["setting"] == "cv" and r["miss_rate"] > 0]
    lines.append(f"- CV 至少一個種子沒命中的 ROI {len(cvr)} 個: 5 個種子都錯 {sum(r['miss_rate'] == 1 for r in cvr)}、"
                 f"3–4 個 {sum(0.5 < r['miss_rate'] < 1 for r in cvr)}、1–2 個 {sum(r['miss_rate'] <= 0.5 for r in cvr)}")
    return rows


def stage1(lines):
    key = {r["alias"]: r for r in csv.DictReader(open(ROOT / "mf_position" / "key.csv", newline=""))}
    fdi = {r["alias"]: int(r["flag_x456"]) for r in csv.DictReader(open(MP / "fdi_flag" / "per_side.csv", newline=""))}
    q = {}
    for pano in sorted({k["pano"] for k in key.values()}):
        img = cv2.imread(str(PANO_DIR / pano), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            q[pano] = global_quality(img)
    sides = [a for a in key if key[a]["pano"] in q and a in fdi]
    flag_axis = {a: int(any(t in key[a]["axis_fallback"] for t in ("unreliable", "no_root", "diff")) or bool(key[a]["reason"])) for a in sides}
    lines.append("\n## 2. 第一階段(牙齒 / FDI): 全景片品質\n")
    lines.append(f"- {len(q)} 張全景片、{len(sides)} 側;兩種編號不一致 {sum(fdi[a] for a in sides)} 側({100 * np.mean([fdi[a] for a in sides]):.1f}%)、參考線保險 {sum(flag_axis.values())} 側")
    lines.append("\n| 特徵 | AUROC 兩種編號不一致 | AUROC 參考線保險 |")
    lines.append("| --- | --- | --- |")
    for ft in ("mean", "std", "range_p5_p95", "sharpness", "noise", "height", "width"):
        a1, _ = auroc([q[key[a]["pano"]][ft] for a in sides], [fdi[a] for a in sides])
        a2, _ = auroc([q[key[a]["pano"]][ft] for a in sides], [flag_axis[a] for a in sides])
        lines.append(f"| {ft} | {a1:.2f} | {a2:.2f} |")
    res = Counter(f"{q[key[a]['pano']]['height']}×{q[key[a]['pano']]['width']}" for a in sides)
    for size, n in res.most_common(4):
        ss = [a for a in sides if f"{q[key[a]['pano']]['height']}×{q[key[a]['pano']]['width']}" == size]
        lines.append(f"- 解析度 {size}: {n} 側,兩種編號不一致 {100 * np.mean([fdi[a] for a in ss]):.1f}%")
    with open(OUT / "stage1_per_pano.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pano"] + list(next(iter(q.values()))))
        w.writeheader()
        w.writerows({"pano": p, **v} for p, v in q.items())


def position(lines):
    rows = [r for r in csv.DictReader(open(MP / "position_seeds" / "per_seed.csv", newline=""))
            if r["variant"].startswith("R34/512 FP32") and r["orig"] in tuple("123456")]
    if not rows:
        return
    wrong = [r for r in rows if r["pred"] != r["orig"]]
    lines.append("\n## 3. 牙位(測試集正常牙,主結果 5 種子,原始標註)\n")
    lines.append(f"- 判錯 {len(wrong)} / {len(rows)}(種子 × 頦孔);沒有預測 {sum(r['pred'] == '' for r in wrong)}")
    mg = [float(r["margin"]) for r in rows if r["margin"] != ""]
    mw = [float(r["margin"]) for r in wrong if r["margin"] != ""]
    lines.append(f"- 邊界距離中位數: 全部 {np.median(mg):.2f} r、判錯 {np.median(mw):.2f} r;判錯中距離 < 0.25 r 的 {sum(m < 0.25 for m in mw)} / {len(mw)}")
    pairs = Counter((r["orig"], r["pred"] or "無") for r in wrong)
    lines.append("- 判錯型態(標註 → 預測): " + "、".join(f"{a}→{b} {n}" for (a, b), n in pairs.most_common(8)))


def main():
    sys.stdout.reconfigure(line_buffering=True)
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# 影像畫質與錯誤分析(quality_errors.py,2026-11-05)\n"]
    stage2(lines)
    stage1(lines)
    position(lines)
    text = "\n".join(lines)
    (OUT / "summary.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
