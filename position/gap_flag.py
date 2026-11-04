"""2026-11-04 用牙齒間隙偵測缺牙(使用者提議;CPU,只讀 items.json)。

10/25–28 試過「用空隙重新編號」(fdi_numbering.py、fdi_tufts.py),在 Tufts 上無效而放棄;這裡只把空隙當「需要人看」的旗標,
不改編號,用測試集的複查後牙位標註(缺牙 21、FDI 錯 8、正常牙 103)驗證。
- 牙齒: items.json 該側(象限)的牙齒輪廓(與標註工具相同),依 FDI 末位排序,只取 x1–x7。
- 特徵(每側取最大值):
    gap   = (下一顆的近心緣 − 這一顆的遠心緣) / 該側牙寬中位數(水平方向;負值 = 重疊)
    space = 相鄰牙中心距離 / 該側相鄰中心距離的中位數(缺一顆約 2)
    count = 該側偵測到的 x1–x7 顆數
- 門檻事先固定: gap ≥ 0.5(空出半顆牙寬)、space ≥ 1.5;另報 AUROC(缺牙 vs 正常牙),屬探索性分析。
- 與 fdi_flag.py 的「兩種編號不一致」取聯集,看合起來能抓多少。
- 輸出 compare_results/mf_position/gap_flag/(per_side.csv、summary.txt)。
"""
import csv
import json
import math
from collections import Counter

import numpy as np

from stage2.compare_yolo_unet import ROOT
from position.mf_position import ITEMS_JSON, KEY_CSV

OUT = ROOT / "compare_results" / "mf_position" / "gap_flag"
LABELS = ROOT / "compare_results" / "mf_position" / "labels_rater_adjudicated.csv"
FDI_FLAG = ROOT / "compare_results" / "mf_position" / "fdi_flag" / "per_side.csv"
GAP_MIN, SPACE_MIN = 0.5, 1.5


def features(item):
    q = int(item["quadrant"])
    teeth = []
    for t in item["teeth"]:
        if len(t["fdi"]) == 2 and t["fdi"][1] in "1234567":
            p = np.asarray(t["poly"], dtype=float)
            teeth.append((int(t["fdi"][1]), p[:, 0].min(), p[:, 0].max(), p[:, 0].mean()))
    teeth.sort()
    if len(teeth) < 3:
        return None
    # 由近心往遠心: 象限 4 在影像左側,遠心 = x 變小;象限 3 遠心 = x 變大。統一成「往遠心為正」
    sign = -1.0 if q == 4 else 1.0
    widths = np.array([t[2] - t[1] for t in teeth])
    w = float(np.median(widths))
    gaps, spaces = [], []
    for a, b in zip(teeth, teeth[1:]):
        if sign > 0:
            gaps.append((b[1] - a[2]) / w)
        else:
            gaps.append((a[1] - b[2]) / w)
        spaces.append(abs(b[3] - a[3]))
    spaces = np.array(spaces) / float(np.median(spaces))
    return {"gap": round(max(gaps), 3), "space": round(float(spaces.max()), 3), "count": len(teeth)}


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def wilson(k, n, z=1.96):
    if n == 0:
        return "—"
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{k}/{n} = {100 * p:.1f}% ({100 * (c - h):.1f}–{100 * (c + h):.1f})"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads(ITEMS_JSON.read_text())
    with open(KEY_CSV, newline="") as f:
        key = {r["alias"]: r for r in csv.DictReader(f)}
    with open(LABELS, newline="", encoding="utf-8-sig") as f:
        lab = {r["alias"]: r for r in csv.DictReader(f)}
    with open(FDI_FLAG, newline="") as f:
        fdi = {r["alias"]: int(r["flag_x456"]) for r in csv.DictReader(f)}
    group = lambda r: "FDI 錯" if r["fdi_wrong"] == "1" else ("缺牙" if r["position"] == "M" else ("正常牙" if r["position"] in list("123456") else None))
    rows = []
    for alias in data["order"]:
        ft = features(data["items"][alias])
        if ft is None:
            continue
        g = group(lab[alias]) if alias in lab else ""
        rows.append({"alias": alias, "set": key[alias]["set"], "group": g or "", **ft, "flag_fdi": fdi.get(alias, "")})
    with open(OUT / "per_side.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    test = [r for r in rows if r["set"] == "test" and r["group"]]
    by = {g: [r for r in test if r["group"] == g] for g in ("缺牙", "FDI 錯", "正常牙")}
    lines = [f"測試集 {len(test)} 個: " + str({g: len(v) for g, v in by.items()})]
    for ft in ("gap", "space", "count"):
        sgn = -1 if ft == "count" else 1  # 顆數越少越可疑
        a1 = auroc([sgn * r[ft] for r in by["缺牙"]], [sgn * r[ft] for r in by["正常牙"]])
        a2 = auroc([sgn * r[ft] for r in by["缺牙"] + by["FDI 錯"]], [sgn * r[ft] for r in by["正常牙"]])
        med = {g: float(np.median([r[ft] for r in v])) for g, v in by.items()}
        lines.append(f"{ft:5s} AUROC 缺牙 vs 正常 {a1:.3f};異常(缺牙+FDI 錯)vs 正常 {a2:.3f};中位數 " + ", ".join(f"{g} {m:.2f}" for g, m in med.items()))
    rules = {f"gap ≥ {GAP_MIN}": lambda r: r["gap"] >= GAP_MIN, f"space ≥ {SPACE_MIN}": lambda r: r["space"] >= SPACE_MIN,
             "兩種編號不一致(fdi_flag)": lambda r: r["flag_fdi"] == 1,
             f"兩種編號不一致 或 gap ≥ {GAP_MIN}": lambda r: r["flag_fdi"] == 1 or r["gap"] >= GAP_MIN,
             f"兩種編號不一致 或 space ≥ {SPACE_MIN}": lambda r: r["flag_fdi"] == 1 or r["space"] >= SPACE_MIN}
    for name, fn in rules.items():
        lines.append(f"\n== {name}")
        for g, v in by.items():
            lines.append(f"  {g:4s} 被標記 {wilson(sum(bool(fn(r)) for r in v), len(v))}")
        ab = by["缺牙"] + by["FDI 錯"]
        lines.append(f"  異常合計 {wilson(sum(bool(fn(r)) for r in ab), len(ab))}")
    lines.append("\n缺牙組的特徵(alias gap space count flag_fdi): " + "; ".join(f"{r['alias']} {r['gap']:.2f} {r['space']:.2f} {r['count']} {r['flag_fdi']}" for r in by["缺牙"]))
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
