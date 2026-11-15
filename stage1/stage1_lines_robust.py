"""2026-11-14 第一階段參考線的穩定度與改善嘗試(使用者要求;伺服器 CPU,探索性;輸出 compare_results/deploy_eval/lines_robust/)

背景: 牙位標註是看著 v2 參考線標的,參考線「對不對」無法用現有標註驗證;能量的是「穩不穩」——同一張片子畫質改變後,
  參考線與牙位答案會不會跟著變(11/13 第一階段換成 INT8 就讓 4.4% 的牙位改變)。11/14 的壓力測試只測了第二階段。
開發集(不看測試集,同 stage1_int8.py): 非測試、有頦孔 GT 的全景片 303 張、476 側;牙位 = GT 頦孔 + 該方法的參考線(中央一半)。
條件(對整張全景片加合成變化,與 robust_stress.py 相同的函式): 雜訊 sd 5 / 10、模糊 sd 1 / 2、對比 ×0.5、gamma 0.6 / 1.6、亮度 ±30、解析度 ×0.5、JPEG q30。
方法:
  v2          現行(OpenVINO FP32 第一階段 + reference_lines_v2)
  v2+flip     水平翻轉再跑一次第一階段,翻回來後兩次的參考線取平均(兩次的 x4–x6 都在才平均,否則用原本的)
  v2+median3  第一階段前對全景片做 3×3 中值濾波(只跑原圖與兩個雜訊條件)
  trim        長軸改用修剪過的主成分(去掉離軸最遠的 10% 像素後重算,兩輪)
  slice       長軸改用「沿軸切片的中心點」擬合(近似中軸線),不受牙冠 / 牙根兩側不對稱影響
指標:
  自我穩定度 = 該方法在變化後的牙位 == 該方法在原圖的牙位(各條件與 11 個條件平均);參考線在頦孔高度的水平位移 p95(px)。
  原圖上與現行 v2 的牙位相同比例(改線的方法一定 < 1;不同的那些側需要重新標註才知道誰對)。
事先定的判準: (1) 不改變原圖答案的方法(v2+flip、v2+median3): 原圖上與 v2 相同 ≥ 99%,且平均穩定度比 v2 高、以全景片 bootstrap 的 95% CI 下界 > 0 → 建議進測試集評分。
  (2) 改線的方法(trim、slice): 只報告;要採用需要關掉參考線重標測試集。
用法: python -m stage1.stage1_lines_robust --shard i --of n(平行跑),全部跑完後 python -m stage1.stage1_lines_robust --summary
"""
import argparse
import csv
import json
import sys
import zlib

import cv2
import numpy as np

import deployment.fast_masks as fast_masks
import position.mf_position as mf_position
from stage2.compare_yolo_unet import ROOT
from stage1.fdi_cropped import assign_fdi_indices, process_and_separate_jaws
from position.mf_position import centroid, classify, distal_normal, reference_lines
from deployment.pipeline_timing import PANO_DIR
from analysis.robust_stress import CONDITIONS
from stage1.stage1_int8 import Stage1, build_ref, dev_set, x_at

OUT = ROOT / "compare_results" / "deploy_eval" / "lines_robust"
CONDS = ["original", "noise sd 5", "noise sd 10", "blur sd 1", "blur sd 2", "contrast x0.5", "gamma 0.6", "gamma 1.6", "brightness -30", "brightness +30",
         "resolution x0.5", "jpeg q30"]
MEDIAN_CONDS = ["original", "noise sd 5", "noise sd 10"]
METHODS = ["v2", "v2+flip", "v2+median3", "trim", "slice"]
PRINCIPAL = mf_position.principal


def principal_trim(pts):
    keep = pts
    for _ in range(2):
        c, v = PRINCIPAL(keep)
        d = np.abs((keep - c) @ np.array([v[1], -v[0]]))
        keep = keep[d <= np.percentile(d, 90)]
        if len(keep) < 20:
            return PRINCIPAL(pts)
    return PRINCIPAL(keep)


def principal_slice(pts):
    c, v = PRINCIPAL(pts)
    t = (pts - c) @ v
    edges = np.linspace(t.min(), t.max() + 1e-6, 13)
    mids = [pts[(t >= a) & (t < b)].mean(axis=0) for a, b in zip(edges[:-1], edges[1:]) if ((t >= a) & (t < b)).sum() >= 5]
    if len(mids) < 4:
        return c, v
    mids = np.array(mids)
    return PRINCIPAL(mids)[0], PRINCIPAL(mids)[1]


def stage1(s1, img):
    _, lower = process_and_separate_jaws(img, s1)
    return assign_fdi_indices(lower) if lower else []


def unflip(lower, w):
    out = []
    for t in lower:
        x1, y1, x2, y2 = t["bbox"]
        q = {3: 4, 4: 3}[t["quadrant"]]
        out.append({**t, "mask": np.ascontiguousarray(t["mask"][:, ::-1]), "bbox": (w - x2, y1, w - x1, y2), "cx": w - 1 - t["cx"], "quadrant": q,
                    "fdi_id": f"{q}{t['fdi_id'][1:]}"})
    return sorted(out, key=lambda t: t["cx"])


def lines_of(lower, q, fn=None):
    mf_position.principal = fn or PRINCIPAL
    try:
        return reference_lines(lower, q, {}) if lower else {}
    finally:
        mf_position.principal = PRINCIPAL


def average_lines(a, b, q):
    if set(a) != set(b) or any(k not in a for k in "456"):
        return a
    out = {}
    for k in a:
        (c1, v1, _), (c2, v2, _) = a[k], b[k]
        p2 = c2 + (c1[1] - c2[1]) / v2[1] * v2
        v = v1 + v2
        v = v / np.linalg.norm(v)
        out[k] = ((c1 + p2) / 2, v, distal_normal(v, q))
    return out


def answer(poly, lines):
    pos, info = classify(poly, lines, "center")
    y = centroid(poly)[1]
    return {"pos": str(pos) if pos else info.get("reason", "none"), "x": {k: round(float(x_at(l, y)), 2) for k, l in lines.items()}}


def run(args):
    fast_masks.install()
    dev = dev_set()
    ref = build_ref(dev)
    panos = sorted(ref)[args.shard::args.of]
    s1 = Stage1()
    rows = []
    for i, pano in enumerate(panos, 1):
        gray = cv2.imread(str(PANO_DIR / pano), cv2.IMREAD_GRAYSCALE)
        w = gray.shape[1]
        sides = [(np.array(p), q) for p, q in ref[pano]["sides"]]
        for cond in CONDS:
            x = CONDITIONS[cond](gray, np.random.default_rng(zlib.crc32(pano.encode())))
            bgr = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
            lower = stage1(s1, bgr)
            lower_f = unflip(stage1(s1, np.ascontiguousarray(bgr[:, ::-1])), w)
            lower_m = stage1(s1, cv2.cvtColor(cv2.medianBlur(x, 3), cv2.COLOR_GRAY2BGR)) if cond in MEDIAN_CONDS else None
            for j, (poly, q) in enumerate(sides):
                base = lines_of(lower, q)
                res = {"v2": answer(poly, base), "v2+flip": answer(poly, average_lines(base, lines_of(lower_f, q), q)),
                       "trim": answer(poly, lines_of(lower, q, principal_trim)), "slice": answer(poly, lines_of(lower, q, principal_slice))}
                if lower_m is not None:
                    res["v2+median3"] = answer(poly, lines_of(lower_m, q))
                for m, a in res.items():
                    rows.append({"pano": pano, "side": j, "condition": cond, "method": m, "pos": a["pos"], "x": json.dumps(a["x"]),
                                 "ref_pos": ref[pano]["res"][j]["pos"]})
        print(f"[{i}/{len(panos)}] {pano}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"rows_{args.shard}of{args.of}.csv", "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0]))
        w_.writeheader()
        w_.writerows(rows)


def summary():
    rows = [r for p in sorted(OUT.glob("rows_*of*.csv")) for r in csv.DictReader(open(p, newline=""))]
    by = {(r["method"], r["condition"], r["pano"], r["side"]): r for r in rows}
    sides = sorted({(r["pano"], r["side"]) for r in rows})
    panos = sorted({p for p, _ in sides})
    lines = ["# 第一階段參考線的穩定度(stage1_lines_robust.py,2026-11-14,探索性)\n", f"- 開發集全景片 {len(panos)} 張、{len(sides)} 側(非測試集)。\n"]
    clean = lambda m, s: by[(m, "original", *s)]
    v2 = {s: clean("v2", s) for s in sides}
    lines.append(f"- 檢查: 現行 v2 在原圖上的牙位與 stage1_int8 存的參考 相同 {np.mean([v2[s]['pos'] == v2[s]['ref_pos'] for s in sides]):.4f}(應為 1)")
    judged = [s for s in sides if v2[s]["pos"].isdigit()]
    lines += ["", "## 1. 原圖上與現行 v2 的差異\n", "| 方法 | 牙位與 v2 相同 | 不同的側數 | 可判定的側數 | 參考線位移 中位數 / p95 px |", "| --- | --- | --- | --- | --- |"]
    out_rows = []
    for m in METHODS:
        same = [clean(m, s)["pos"] == v2[s]["pos"] for s in sides]
        sh = [abs(json.loads(clean(m, s)["x"])[k] - xv) for s in sides for k, xv in json.loads(v2[s]["x"]).items() if k in json.loads(clean(m, s)["x"])]
        lines.append(f"| {m} | {np.mean(same):.4f} | {len(same) - sum(same)} | {sum(clean(m, s)['pos'].isdigit() for s in sides)} | {np.median(sh):.1f} / {np.percentile(sh, 95):.1f} |")
        out_rows.append({"table": "clean_vs_v2", "method": m, "condition": "original", "value": float(np.mean(same)), "changed": len(same) - sum(same)})
    conds = [c for c in CONDS if c != "original"]
    lines += ["", "## 2. 自我穩定度: 變化後的牙位 == 同一方法在原圖的牙位\n", "| 條件 | " + " | ".join(METHODS) + " |", "| --- |" + " --- |" * len(METHODS)]
    stab = {m: {} for m in METHODS}
    for c in conds:
        cells = []
        for m in METHODS:
            if (m, c, *sides[0]) not in by:
                cells.append("—")
                continue
            v = {s: float(by[(m, c, *s)]["pos"] == clean(m, s)["pos"]) for s in sides}
            stab[m][c] = v
            cells.append(f"{np.mean(list(v.values())):.4f}")
            out_rows.append({"table": "stability", "method": m, "condition": c, "value": float(np.mean(list(v.values()))), "changed": int(len(v) - sum(v.values()))})
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    rng = np.random.default_rng(0)

    def mean_stab(m, cs):
        return {s: float(np.mean([stab[m][c][s] for c in cs])) for s in sides}

    def boot(d):
        per = {p: [0.0, 0] for p in panos}
        for (p, _), v in d.items():
            per[p][0] += v
            per[p][1] += 1
        arr = np.array(list(per.values()))
        idx = rng.integers(0, len(arr), (2000, len(arr)))
        st = arr[idx][:, :, 0].sum(1) / arr[idx][:, :, 1].sum(1)
        return np.percentile(st, [2.5, 97.5])

    lines += ["", "## 3. 平均穩定度與現行 v2 的差(以全景片 bootstrap)\n", "| 方法 | 條件 | 平均穩定度 | v2 | 差 [95% CI] |", "| --- | --- | --- | --- | --- |"]
    for m in METHODS[1:]:
        cs = [c for c in conds if c in stab[m]]
        a, b = mean_stab(m, cs), mean_stab("v2", cs)
        d = {s: a[s] - b[s] for s in sides}
        lo, hi = boot(d)
        lines.append(f"| {m} | {len(cs)} 個條件 | {np.mean(list(a.values())):.4f} | {np.mean(list(b.values())):.4f} | {100 * np.mean(list(d.values())):+.2f} 點 [{100 * lo:+.2f}, {100 * hi:+.2f}] |")
        out_rows.append({"table": "mean_stability_diff", "method": m, "condition": f"{len(cs)} conditions", "value": float(np.mean(list(d.values()))), "lo": float(lo), "hi": float(hi)})
    # 參考線位移
    lines += ["", "## 4. 參考線在頦孔高度的水平位移 p95(px,對同一方法的原圖)\n", "| 條件 | " + " | ".join(METHODS) + " |", "| --- |" + " --- |" * len(METHODS)]
    for c in conds:
        cells = []
        for m in METHODS:
            if c not in stab[m]:
                cells.append("—")
                continue
            sh = [abs(xv - json.loads(by[(m, c, *s)]["x"])[k]) for s in sides for k, xv in json.loads(clean(m, s)["x"]).items() if k in json.loads(by[(m, c, *s)]["x"])]
            cells.append(f"{np.percentile(sh, 95):.1f}")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    (OUT / "summary.md").write_text(text + "\n", encoding="utf-8")
    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in out_rows for k in r)))
        w.writeheader()
        w.writerows(out_rows)
    print(text)


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--of", type=int, default=1)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    summary() if args.summary else run(args)
