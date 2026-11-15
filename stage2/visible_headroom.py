"""2026-11-14 可辨識影像上還有多少進步空間(使用者提問;CPU,只讀已存遮罩)。輸出 compare_results/quality_gate/visible_headroom.md"""
import csv
from collections import defaultdict

import numpy as np

from stage2.compare_yolo_unet import ROOT
from stage2.quality_gate import OUT, load_items

items = load_items()
lines = ["# 可辨識影像上的進步空間(visible_headroom.py;預測 = 5 種子多數決 lcc(CV)/ 25 個模型多數決 lcc(測試))", "",
         "| 資料 | 分層 | n | 命中 1R | 命中 0.5R | 沒命中數 | 中心誤差 / R 中位數 | p90 | mf Dice 平均 | Dice 中位數 | Dice < 0.7 的比例 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
for setting in ("cv", "test"):
    for name, v in (("可辨識", 1), ("無法辨識", 0)):
        sub = [t for t in items if t["setting"] == setting and t["visible"] == v]
        err, dice = [], []
        for t in sub:
            p, g = t["pred"], t["gt"]
            dice.append(2 * (p & g).sum() / (p.sum() + g.sum()))
            err.append(np.linalg.norm(np.argwhere(p).mean(0) - t["c"]) / t["R"] if p.any() else np.inf)
        err, dice = np.array(err), np.array(dice)
        lines.append(f"| {setting} | {name} | {len(sub)} | {np.mean(err <= 1):.3f} | {np.mean(err <= 0.5):.3f} | {int((err > 1).sum())} | {np.median(err):.2f} | {np.percentile(err, 90):.2f} | "
                     f"{dice.mean():.3f} | {np.median(dice):.3f} | {np.mean(dice < 0.7):.3f} |")
        if v == 1:
            miss = [(t["stem"], round(float(e), 2), round(float(t["u"]), 2)) for t, e in zip(sub, err) if e > 1]
            lines.append(f"|  | 可辨識但沒命中(stem, 誤差 / R, 分歧度) | | | | | | | | | {miss} |")
# 標註者自身一致性(同一人兩次標註的 mf Dice)
ag = [float(r["old_vs_new"]) for r in csv.DictReader(open(ROOT / "annotation_agreement" / "agreement.csv", newline="")) if r["class"] == "mf" and float(r["old_area"]) > 0 and float(r["new_area"]) > 0]
lines += ["", f"- 標註者自身一致性(兩次都有畫頦孔的 {len(ag)} 個 ROI): mf Dice 平均 {np.mean(ag):.3f}、中位數 {np.median(ag):.3f}"]
# 牙位: 測試集正常牙,依可辨識度分層(主結果 5 種子)
vis = {r["stem"]: int(r["mf_visible"]) for r in csv.DictReader(open(ROOT / "visibility_grading" / "visibility.csv", newline=""))}
key = {r["alias"]: r["stem"] for r in csv.DictReader(open(ROOT / "mf_position" / "key.csv", newline=""))}
MP = ROOT / "compare_results" / "mf_position"
for lab_name, path in (("原始標註", ROOT / "mf_position/labels_rater.csv"), ("複查後標註", MP / "labels_adjudicated_dir/labels_rater.csv")):
    lab = {r["alias"]: r["position"] for r in csv.DictReader(open(path, newline="", encoding="utf-8-sig"))}
    acc = defaultdict(list)
    for r in csv.DictReader(open(MP / "position_seeds/per_seed.csv", newline="")):
        if r["variant"] == "R34/512 FP32 五折(主結果)" and lab.get(r["alias"]) in tuple("123456") and key.get(r["alias"]) in vis:
            acc[(vis[key[r["alias"]]], r["alias"])].append(r["pred"] == lab[r["alias"]])
    for v, name in ((1, "可辨識"), (0, "無法辨識")):
        a = [np.mean(x) for (vv, _), x in acc.items() if vv == v]
        lines.append(f"- 牙位(測試集正常牙,{lab_name},主結果 5 種子平均){name}: n = {len(a)},正確 {100 * np.mean(a):.1f}%")
text = "\n".join(lines)
(OUT / "visible_headroom.md").write_text(text + "\n", encoding="utf-8")
print(text)
