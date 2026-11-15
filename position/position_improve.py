"""2026-11-14 牙位判定的改善嘗試(使用者要求;CPU,只讀已存遮罩;輸出 compare_results/mf_position/improve/)

背景: 規則套在 GT 頦孔上與複查後標註 100% 相同,所以正常牙剩下的牙位錯誤都來自「預測頦孔和 GT 頦孔的差異跨過判定邊界」(約三成頦孔離邊界 < 0.5 R)。
這裡不動參考線(標註是看著 v2 參考線標的,改線無法用現有標註驗證),只改「怎麼從預測頦孔決定牙位」。
開發指標(不需要人工牙位標註): CV 的預測頦孔牙位 == 規則套在 GT 頦孔上的牙位(key.csv auto_position_center),5 種子。
測試集(原始 / 複查後標註,正常牙,5 種子五折)只做確認。
診斷: 預測頦孔在參考線法向的半寬 / GT 半寬、質心沿法向的有號位移(正 = 遠心)、牙位改變的方向。
版本(門檻不依測試集調):
  V0 現行: 遮罩 → lcc → 中央一半(|d| ≤ 0.5 × 預測半寬)
  V1 半寬校正: |d| ≤ 0.5 × s × 預測半寬,s = CV 上「GT 半寬 / 預測半寬」的中位數(只用命中的 ROI;評 CV 的第 k 折時 s 只用其他折估)
  V2 質心校正: 質心沿法向減去 CV 的中位數位移(以預測半寬為單位),同樣其他折估
  V3 牙位投票: 各成員(CV = 5 個種子、測試 = 五折)各自判牙位再多數決,平手用 V0 的答案
  V4 平均質心與半寬: 各成員的質心、半寬取平均再判
  CV 的 V3 / V4 以「5 個種子」當成員,對照是 5 個種子的遮罩多數決(E0),只用來看方向;測試集的成員是同一個種子的五折(部署時拿得到)。
事先定的判準: CV 一致率的差(以全景片為單位 bootstrap)95% CI 下界 > 0,且測試集兩種標註都不變差,才建議採用。
"""
import csv
import json
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

import position.mf_position_eval as ev
from stage2.compare_yolo_unet import ROOT
from position.mf_position import ITEMS_JSON, KEY_CSV, OUT as MPOS, centroid
from analysis.paper_metrics import read_masks
from position.position_seeds import boot

MAIN = ROOT / ev.MASKS
DST = ROOT / "compare_results" / "mf_position" / "improve"
TAGS = ["r34_imagenet_aug"] + [f"r34_imagenet_aug_s{s}" for s in range(1, 5)]


def poly_of(mask):
    if mask is None or not mask.any():
        return None
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cs, _ = cv2.findContours((lab == big).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float)


def geom(poly, lines):
    # 質心、各參考線的有號距離 d 與法向半寬 half
    m = centroid(poly)
    return m, {k: float((m - c) @ n) for k, (c, _, n) in lines.items()}, {k: float(np.ptp((poly - m) @ n)) / 2 for k, (_, _, n) in lines.items()}


def decide(dist, half, frac=0.5, scale=1.0, shift=0.0):
    # 與 mf_position.classify(rule="center") 相同;scale 放大半寬、shift(半寬為單位,正 = 遠心)從質心扣掉
    if any(k not in dist for k in "456"):
        return ""
    d = {k: dist[k] - shift * half[k] for k in dist}
    crossing = [k for k in d if abs(d[k]) <= frac * scale * half[k]]
    if crossing:
        return str({"4": 2, "5": 4, "6": 6}[min(crossing, key=lambda k: abs(d[k]))])
    return "1" if d["4"] < 0 else "3" if d["5"] < 0 else "5" if d["6"] < 0 else "6"


def vote_pos(positions, fallback):
    c = Counter(p for p in positions if p)
    if not c:
        return fallback
    top = c.most_common()
    return top[0][0] if len(top) == 1 or top[0][1] > top[1][1] else fallback


def main():
    sys.stdout.reconfigure(line_buffering=True)
    DST.mkdir(parents=True, exist_ok=True)
    items = json.loads(ITEMS_JSON.read_text(encoding="utf-8"))["items"]
    keys = list(csv.DictReader(open(KEY_CSV, newline="")))
    sources = {r["stem"]: r for r in csv.DictReader(open(ev.ROI_SOURCES, newline=""))}
    folds = {r["stem"]: int(r["fold"]) for r in csv.DictReader(open(MAIN / "folds.csv", newline=""))}
    orig = {r["alias"]: r["position"] for r in csv.DictReader(open(MPOS / "labels_rater.csv", newline="", encoding="utf-8-sig"))}
    adj = {r["alias"]: r["position"] for r in csv.DictReader(open(ROOT / "compare_results/mf_position/labels_adjudicated_dir/labels_rater.csv", newline="", encoding="utf-8-sig"))}

    recs = []  # 每個 (側, 種子) 的幾何量
    for k in keys:
        item, src = items[k["alias"]], sources[k["stem"]]
        cx, cy = int(k["crop_x"]), int(k["crop_y"])
        lines = ev.lines_from_item(item, cx, cy, "a")
        if any(q not in lines for q in "456"):
            continue
        off = (int(src["x"]), int(src["y"]))
        gt_poly = np.array(item["mf"]) + (cx, cy)
        gm, gd, gh = geom(gt_poly, lines)
        R = (cv2.contourArea(gt_poly.astype(np.float32)) / np.pi) ** 0.5
        test = k["set"] == "test"
        if not test and k["stem"] not in folds:
            continue
        base = dict(alias=k["alias"], pano=k["pano"], set=k["set"], fold=folds.get(k["stem"], -1), rule_gt=str(k["auto_position_center"]), gd=gd, gh=gh)

        def member(mask):
            poly = poly_of(mask)
            if poly is None:
                return None
            m, d, h = geom(poly + off, lines)
            return dict(m=m, d=d, h=h, hit=float(np.linalg.norm(m - gm) <= R))

        seed_masks = []
        for s, tag in enumerate(TAGS):
            if test:
                ms = [read_masks(MAIN, "test", tag, f, k["stem"]) for f in range(5)]
                if any(x is None for x in ms):
                    break
                ms = [x["mf"] for x in ms]
                vote = np.sum(ms, axis=0) > 2.5
                recs.append({**base, "seed": s, "main": member(vote), "members": [member(x) for x in ms]})
            else:
                x = read_masks(MAIN, "cv", tag, folds[k["stem"]], k["stem"])
                if x is None:
                    break
                seed_masks.append(x["mf"])
                recs.append({**base, "seed": s, "main": member(x["mf"]), "members": None})
        if not test and len(seed_masks) == 5:
            recs.append({**base, "seed": "ens", "main": member(np.sum(seed_masks, axis=0) > 2.5), "members": [member(x) for x in seed_masks]})

    lines_out = ["# 牙位判定的改善嘗試(position_improve.py,2026-11-14)\n"]
    # ---- 診斷 ----
    lines_out += ["## 1. 診斷: 預測頦孔相對 GT 頦孔(沿最近參考線的法向;只看命中的 ROI)\n", "| 資料 | n | 預測半寬 / GT 半寬 中位數 [四分位] | 質心位移 / 預測半寬 中位數(正 = 遠心) | |位移| 中位數 |", "| --- | --- | --- | --- | --- |"]

    def ratios(sub):
        out = []
        for r in sub:
            m = r["main"]
            if m and m["hit"]:
                k = min("456", key=lambda q: abs(r["gd"][q]))
                out.append((m["h"][k] / max(r["gh"][k], 1e-6), (m["d"][k] - r["gd"][k]) / max(m["h"][k], 1e-6)))
        return np.array(out)

    groups = {"CV 單一模型(5 種子)": [r for r in recs if r["set"] != "test" and r["seed"] != "ens"], "CV 5 種子多數決": [r for r in recs if r["seed"] == "ens"],
              "測試 五折多數決(5 種子)": [r for r in recs if r["set"] == "test"]}
    for name, sub in groups.items():
        a = ratios(sub)
        q = np.percentile(a[:, 0], [25, 50, 75])
        lines_out.append(f"| {name} | {len(a)} | {q[1]:.3f} [{q[0]:.3f}, {q[2]:.3f}] | {np.median(a[:, 1]):+.3f} | {np.median(np.abs(a[:, 1])):.3f} |")
    cv = groups["CV 單一模型(5 種子)"]
    changed = Counter((r["rule_gt"], decide(r["main"]["d"], r["main"]["h"])) for r in cv if r["main"] and r["rule_gt"])
    diff = sorted(((a, b, c) for (a, b), c in changed.items() if a != b), key=lambda t: -t[2])
    lines_out += ["", "- CV 牙位改變的方向(GT 頦孔的牙位 → 預測頦孔的牙位,5 種子合計): " + ", ".join(f"{a}→{b} ×{c}" for a, b, c in diff[:10])]
    hit_changed = sum(c for (a, b), c in changed.items() if a != b)
    with_hit = Counter((r["rule_gt"] != decide(r["main"]["d"], r["main"]["h"]), bool(r["main"]["hit"])) for r in cv if r["main"] and r["rule_gt"])
    lines_out.append(f"- CV 牙位改變 {hit_changed} 次,其中頦孔有命中 {with_hit[(True, True)]}、沒命中 {with_hit[(True, False)]};"
                     f"另有沒有預測頦孔 {sum(1 for r in cv if not r['main'] and r['rule_gt'])} 次")

    # ---- 校正量(其他折估) ----
    def calib(sub):
        a = ratios(sub)
        return float(np.median(1 / a[:, 0])), float(np.median(a[:, 1]))

    cal_single = {f: calib([r for r in cv if r["fold"] != f]) for f in range(5)}
    cal_single[-1] = calib(cv)
    ens = groups["CV 5 種子多數決"]
    cal_ens = {f: calib([r for r in ens if r["fold"] != f]) for f in range(5)}
    cal_ens[-1] = calib(ens)
    lines_out.append(f"- 校正量(全部 CV): 單一模型 s = {cal_single[-1][0]:.3f}、位移 = {cal_single[-1][1]:+.3f};多數決遮罩 s = {cal_ens[-1][0]:.3f}、位移 = {cal_ens[-1][1]:+.3f}")

    def answers(r, cal):
        m = r["main"]
        if m is None:
            return {v: "" for v in ("V0", "V1", "V2", "V1+V2", "V3", "V4")}
        s, sh = cal[r["fold"] if r["set"] != "test" else -1]
        out = {"V0": decide(m["d"], m["h"]), "V1": decide(m["d"], m["h"], scale=s), "V2": decide(m["d"], m["h"], shift=sh), "V1+V2": decide(m["d"], m["h"], scale=s, shift=sh)}
        mem = [x for x in (r["members"] or []) if x]
        if mem:
            out["V3"] = vote_pos([decide(x["d"], x["h"]) for x in mem], out["V0"])
            out["V4"] = decide({k: float(np.mean([x["d"][k] for x in mem])) for k in "456"}, {k: float(np.mean([x["h"][k] for x in mem])) for k in "456"})
        else:
            out["V3"] = out["V4"] = ""
        return out

    rows = []

    def table(title, sub, cal, target, variants):
        # target(r) → 正確答案;每側先對種子平均,以全景片 bootstrap
        acc = {v: defaultdict(list) for v in variants}
        pano = {}
        for r in sub:
            t = target(r)
            if not t:
                continue
            a = answers(r, cal)
            pano[r["alias"]] = r["pano"]
            for v in variants:
                acc[v][r["alias"]].append(a[v] == t)
        lines_out.extend(["", f"### {title}(n = {len(acc[variants[0]])} 側)", "", "| 版本 | 一致 / 正確率 | 與 V0 的差 [95% CI] | p |", "| --- | --- | --- | --- |"])
        base = {a: np.mean(x) for a, x in acc[variants[0]].items()}
        for v in variants:
            cur = {a: np.mean(x) for a, x in acc[v].items()}
            if v == variants[0]:
                cell = "— | —"
                d = lo = hi = p = 0.0
            else:
                pp = defaultdict(lambda: [0.0, 0])
                for a in cur:
                    pp[pano[a]][0] += cur[a] - base[a]
                    pp[pano[a]][1] += 1
                lo, hi, p = boot(list(pp.values()))
                d = float(np.mean([cur[a] - base[a] for a in cur]))
                cell = f"{100 * d:+.2f} 點 [{100 * lo:+.2f}, {100 * hi:+.2f}] | {p:.2f}"
            lines_out.append(f"| {v} | {100 * np.mean(list(cur.values())):.2f}% | {cell} |")
            rows.append({"table": title, "variant": v, "n": len(cur), "acc": float(np.mean(list(cur.values()))), "diff": d, "lo": float(lo), "hi": float(hi), "p": float(p)})

    lines_out.append("\n## 2. 結果\n")
    names = {"V0": "V0 現行", "V1": "V1 半寬校正", "V2": "V2 質心校正", "V1+V2": "V1 + V2", "V3": "V3 牙位投票", "V4": "V4 平均質心與半寬"}
    table("CV 單一模型,5 種子: 預測牙位 == GT 頦孔的牙位", cv, cal_single, lambda r: r["rule_gt"], ["V0", "V1", "V2", "V1+V2"])
    table("CV 5 個種子當成員: 預測牙位 == GT 頦孔的牙位", ens, cal_ens, lambda r: r["rule_gt"], ["V0", "V1", "V2", "V1+V2", "V3", "V4"])
    test = groups["測試 五折多數決(5 種子)"]
    for lab, name in ((orig, "原始標註"), (adj, "複查後標註")):
        table(f"測試集正常牙,五折,5 種子: 對{name}", test, cal_ens, lambda r, lab=lab: lab.get(r["alias"], "") if lab.get(r["alias"], "") in tuple("123456") else "", ["V0", "V1", "V2", "V1+V2", "V3", "V4"])
    table("測試集,五折,5 種子: 預測牙位 == GT 頦孔的牙位(不用人工標註)", test, cal_ens, lambda r: r["rule_gt"], ["V0", "V1", "V2", "V1+V2", "V3", "V4"])
    lines_out += ["", "版本: " + ";".join(f"{k} = {v}" for k, v in names.items())]
    text = "\n".join(lines_out)
    (DST / "summary.md").write_text(text + "\n", encoding="utf-8")
    with open(DST / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(text)


if __name__ == "__main__":
    main()
