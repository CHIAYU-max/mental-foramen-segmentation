"""2026-11-14 第一階段不穩定時,轉介旗標擋不擋得住?再加一個「翻轉一致性」旗標有沒有用(使用者要求;伺服器 CPU,探索性)

stage1_lines_robust.py / lines_breakdown.py: 畫質小幅改變就讓 4–11% 的牙位改變,其中 77% 是整條參考線跳到別顆牙或不見(牙齒偵測 / 編號改變),
  只有 23% 是線小幅移動。改長軸的算法、翻轉平均都沒有幫助。所以這裡不再改線,改看「編號不穩時系統知不知道」:
  現行旗標(11/4 定案,deploy_pipeline.stage1): 兩種編號在 x4–x6 任一不一致,或頦孔側比對側少 ≥ 1 顆 → 轉介。
  新旗標(翻轉一致性): 把全景片水平翻轉再跑一次第一階段,兩次結果不一致就轉介。兩種定義:
    flipJ  兩次的 x4–x6 參考線有任一條不見,或在頦孔高度相差 > 30 px(相鄰參考線間距中位數 125 px)—— 不需要頦孔位置,只抓「換了一顆牙」
    flipP  兩次的牙位答案不同(用 GT 頦孔;部署時用預測頦孔)
開發集: 同 stage1_lines_robust.py(303 張、476 側,不看測試集);牙位 = GT 頦孔 + 當次的參考線。條件相同(11 種合成變化 + 原圖)。
指標: 每個條件下 牙位改變率、靜默改變率(牙位和原圖不同、又沒有被旗標轉介)、轉介率;原圖上各旗標的轉介率。
事先定的判準: 加上新旗標後,11 個條件平均的靜默改變率下降、以全景片 bootstrap 的 95% CI 下界 > 0,且原圖上多轉介的側 ≤ 5 點,才進測試集做系統層級評分。
用法: --shard i --of n 平行跑,之後 --summary。輸出 compare_results/deploy_eval/lines_robust/flag_rows_*.csv、flag_summary.md
"""
import argparse
import csv
import json
import sys
import zlib

import cv2
import numpy as np

import deployment.deploy_pipeline as D
import deployment.fast_masks as fast_masks
from stage2.compare_yolo_unet import ROOT
from position.mf_position import centroid, classify
from deployment.pipeline_timing import PANO_DIR
from analysis.robust_stress import CONDITIONS
from stage1.stage1_int8 import build_ref, dev_set, x_at
from stage1.stage1_lines_robust import CONDS

OUT = ROOT / "compare_results" / "deploy_eval" / "lines_robust"
JUMP = 30.0


def answer(poly, lines):
    pos, info = classify(poly, lines, "center") if lines else (None, {"reason": "none"})
    y = centroid(poly)[1]
    return (str(pos) if pos else info.get("reason", "none")), {k: round(float(x_at(l, y)), 2) for k, l in lines.items()}


def run(args):
    fast_masks.install()
    D.T32_IMGSZ = 640
    teeth, model32 = D.load_yolo("ov_mixed")
    ref = build_ref(dev_set())
    panos = sorted(ref)[args.shard::args.of]
    rows = []
    for i, pano in enumerate(panos, 1):
        gray = cv2.imread(str(PANO_DIR / pano), cv2.IMREAD_GRAYSCALE)
        w = gray.shape[1]
        sides_gt = [(np.array(p), q) for p, q in ref[pano]["sides"]]
        for cond in CONDS:
            x = CONDITIONS[cond](gray, np.random.default_rng(zlib.crc32(pano.encode())))
            bgr = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
            _, _, sides = D.stage1(bgr, teeth, model32, D.Clock())
            _, _, sides_f = D.stage1(np.ascontiguousarray(bgr[:, ::-1]), teeth, model32, D.Clock())
            for j, (poly, q) in enumerate(sides_gt):
                lines, flag_fdi, flag_side, _ = sides[q]
                pos, xs = answer(poly, lines)
                poly_f = poly.copy()
                poly_f[:, 0] = w - 1 - poly_f[:, 0]
                lines_f, ff_fdi, ff_side, _ = sides_f[{3: 4, 4: 3}[q]]
                pos_f, xs_f = answer(poly_f, lines_f)
                rows.append({"pano": pano, "side": j, "condition": cond, "pos": pos, "x": json.dumps(xs), "flag_fdi": flag_fdi, "flag_side": flag_side,
                             "pos_f": pos_f, "x_f": json.dumps({k: round(w - 1 - v, 2) for k, v in xs_f.items()}), "flag_fdi_f": ff_fdi, "flag_side_f": ff_side})
        print(f"[{i}/{len(panos)}] {pano}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"flag_rows_{args.shard}of{args.of}.csv", "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0]))
        w_.writeheader()
        w_.writerows(rows)


def summary():
    rows = [r for p in sorted(OUT.glob("flag_rows_*of*.csv")) for r in csv.DictReader(open(p, newline=""))]
    by = {(r["condition"], r["pano"], r["side"]): r for r in rows}
    sides = sorted({(r["pano"], r["side"]) for r in rows})
    panos = sorted({p for p, _ in sides})
    conds = [c for c in CONDS if c != "original"]

    def flags(r):
        xa, xb = json.loads(r["x"]), json.loads(r["x_f"])
        complete = all(k in xa for k in "456") and all(k in xb for k in "456")
        exist = r["flag_fdi"] == "1" or r["flag_side"] == "1" or not r["pos"].isdigit()
        flip_j = (not complete) or max(abs(xa[k] - xb[k]) for k in "456") > JUMP
        flip_p = r["pos"] != r["pos_f"]
        return {"none": False, "現行旗標": exist, "現行 + flipJ": exist or flip_j, "現行 + flipP": exist or flip_p, "現行 + flipJ + flipP": exist or flip_j or flip_p,
                "只有 flipJ": flip_j, "只有 flipP": flip_p}

    names = ["none", "現行旗標", "現行 + flipJ", "現行 + flipP", "現行 + flipJ + flipP"]
    lines = ["# 第一階段不穩定時的轉介旗標(stage1_flag_robust.py,2026-11-14,探索性)\n", f"- 開發集全景片 {len(panos)} 張、{len(sides)} 側。\n", "## 1. 原圖上的轉介率\n"]
    clean = {s: flags(by[("original", *s)]) for s in sides}
    for n in ["現行旗標", "只有 flipJ", "只有 flipP", "現行 + flipJ", "現行 + flipP", "現行 + flipJ + flipP"]:
        lines.append(f"- {n}: {np.mean([clean[s][n] for s in sides]):.1%}")
    lines += ["", "## 2. 各條件: 牙位改變率、轉介率與靜默改變率(牙位和原圖不同、又沒被轉介)\n",
              "| 條件 | 牙位改變 | 轉介率 現行 → +flipJ → +flipP → 兩者 | 靜默改變: 不轉介 | 現行旗標 | + flipJ | + flipP | + 兩者 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    silent = {n: {} for n in names}
    out = []
    for c in conds:
        ch = {s: by[(c, *s)]["pos"] != by[("original", *s)]["pos"] for s in sides}
        fl = {s: flags(by[(c, *s)]) for s in sides}
        for n in names:
            silent[n][c] = {s: float(ch[s] and not fl[s][n]) for s in sides}
        ref_rate = [np.mean([fl[s][n] for s in sides]) for n in names[1:]]
        lines.append(f"| {c} | {np.mean(list(ch.values())):.1%} | " + " → ".join(f"{r:.1%}" for r in ref_rate) + " | "
                     + " | ".join(f"{np.mean(list(silent[n][c].values())):.1%}" for n in names) + " |")
        out.append({"condition": c, "changed": float(np.mean(list(ch.values()))), **{f"refer|{n}": float(np.mean([fl[s][n] for s in sides])) for n in names[1:]},
                    **{f"silent|{n}": float(np.mean(list(silent[n][c].values()))) for n in names}})
    rng = np.random.default_rng(0)

    def boot(d):
        per = {p: [0.0, 0] for p in panos}
        for (p, _), v in d.items():
            per[p][0] += v
            per[p][1] += 1
        arr = np.array(list(per.values()))
        idx = rng.integers(0, len(arr), (2000, len(arr)))
        return np.percentile(arr[idx][:, :, 0].sum(1) / arr[idx][:, :, 1].sum(1), [2.5, 97.5])

    lines += ["", "## 3. 11 個條件平均的靜默改變率(差 = 對現行旗標,以全景片 bootstrap)\n", "| 旗標 | 平均靜默改變率 | 與現行旗標的差 [95% CI] | 原圖轉介率 |", "| --- | --- | --- | --- |"]
    mean = {n: {s: float(np.mean([silent[n][c][s] for c in conds])) for s in sides} for n in names}
    for n in names:
        d = {s: mean[n][s] - mean["現行旗標"][s] for s in sides}
        lo, hi = boot(d)
        cell = "—" if n == "現行旗標" else f"{100 * np.mean(list(d.values())):+.2f} 點 [{100 * lo:+.2f}, {100 * hi:+.2f}]"
        lines.append(f"| {n} | {np.mean(list(mean[n].values())):.2%} | {cell} | {np.mean([clean[s][n] for s in sides]):.1%} |")
        out.append({"condition": "mean", "flag": n, "silent": float(np.mean(list(mean[n].values()))), "diff": float(np.mean(list(d.values()))), "lo": float(lo), "hi": float(hi),
                    "clean_refer": float(np.mean([clean[s][n] for s in sides]))})
    text = "\n".join(lines)
    (OUT / "flag_summary.md").write_text(text + "\n", encoding="utf-8")
    with open(OUT / "flag_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in out for k in r)))
        w.writeheader()
        w.writerows(out)
    print(text)


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--of", type=int, default=1)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    summary() if args.summary else run(args)
