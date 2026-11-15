"""2026-11-14 參考線不穩定的原因拆解(讀 stage1_lines_robust.py 的 rows_*.csv):牙位改變時,是整條線跳到別顆牙(編號 / 偵測改變),還是線只小幅移動、剛好跨過判定邊界。"""
import csv
import json
from collections import Counter, defaultdict

import numpy as np

from stage2.compare_yolo_unet import ROOT

OUT = ROOT / "compare_results" / "deploy_eval" / "lines_robust"
rows = [r for p in sorted(OUT.glob("rows_*of*.csv")) for r in csv.DictReader(open(p, newline=""))]
by = {(r["method"], r["condition"], r["pano"], r["side"]): r for r in rows}
sides = sorted({(r["pano"], r["side"]) for r in rows})
conds = [c for c in dict.fromkeys(r["condition"] for r in rows) if c != "original"]
JUMP = 30.0  # px;參考線在頦孔高度移動超過這個量視為換了一顆牙(相鄰牙的參考線間距中位數見下)
gaps = []
for s in sides:
    x = json.loads(by[("v2", "original", *s)]["x"])
    if all(k in x for k in "456"):
        gaps += [abs(x["5"] - x["4"]), abs(x["6"] - x["5"])]
lines = ["# 參考線不穩定的原因(lines_breakdown.py)", "", f"- 原圖上相鄰參考線在頦孔高度的間距: 中位數 {np.median(gaps):.0f} px(p10 {np.percentile(gaps, 10):.0f});「跳線」門檻 {JUMP:.0f} px", "",
         "| 條件 | 牙位改變 | 其中: 參考線不完整(缺牙 / 沒偵測到) | 整條線跳動 > 30 px(編號或偵測改變) | 線小幅移動、跨過判定邊界 | 小幅移動時的位移中位數 px |", "| --- | --- | --- | --- | --- | --- |"]
tot = Counter()
out = []
for c in conds:
    cnt, small = Counter(), []
    for s in sides:
        a, b = by[("v2", "original", *s)], by[("v2", c, *s)]
        if a["pos"] == b["pos"]:
            continue
        xa, xb = json.loads(a["x"]), json.loads(b["x"])
        if not b["pos"].isdigit() or not a["pos"].isdigit():
            kind = "missing"
        else:
            sh = max(abs(xa[k] - xb[k]) for k in xa if k in xb)
            kind = "jump" if sh > JUMP else "small"
            if kind == "small":
                small.append(sh)
        cnt[kind] += 1
    n = sum(cnt.values())
    tot.update(cnt)
    lines.append(f"| {c} | {n}({n / len(sides):.1%}) | {cnt['missing']} | {cnt['jump']} | {cnt['small']} | {np.median(small) if small else 0:.1f} |")
    out.append({"condition": c, "changed": n, "missing": cnt["missing"], "jump": cnt["jump"], "small": cnt["small"]})
n = sum(tot.values())
lines.append(f"| 合計 | {n} | {tot['missing']}({tot['missing'] / n:.0%}) | {tot['jump']}({tot['jump'] / n:.0%}) | {tot['small']}({tot['small'] / n:.0%}) | |")
# 原圖上離判定邊界多近的側比較容易變
text = "\n".join(lines)
(OUT / "breakdown.md").write_text(text + "\n", encoding="utf-8")
with open(OUT / "breakdown.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
print(text)
