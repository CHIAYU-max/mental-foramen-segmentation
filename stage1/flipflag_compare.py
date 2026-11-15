"""2026-11-14 翻轉一致性旗標在測試集的系統層級配對比較(讀 deploy_eval/score_t32_640 與 score_flipflag 的 per_side.csv)。"""
import csv
from collections import defaultdict

import numpy as np

from stage2.compare_yolo_unet import ROOT
from position.position_referral import group
from position.position_seeds import boot

BASE = ROOT / "compare_results" / "deploy_eval"
MP = ROOT / "compare_results" / "mf_position"
labs = {"原始標註": {r["alias"]: r for r in csv.DictReader(open(ROOT / "mf_position/labels_rater.csv", newline="", encoding="utf-8-sig"))},
        "複查後標註": {r["alias"]: r for r in csv.DictReader(open(MP / "labels_adjudicated_dir/labels_rater.csv", newline="", encoding="utf-8-sig"))}}
adj = labs["複查後標註"]
runs = {n: list(csv.DictReader(open(BASE / n / "per_side.csv", newline="", encoding="utf-8"))) for n in ("score_t32_640", "score_flipflag")}
labeled = sorted({r["alias"] for r in runs["score_t32_640"]} & {a for a, r in adj.items() if r.get("position") or r.get("fdi_wrong") == "1"})
grp = {a: group(adj[a]) for a in labeled}
pano = {r["alias"]: r["pano"] for r in runs["score_t32_640"]}
lines = ["# 翻轉一致性旗標: 測試集系統層級(flipflag_compare.py;5 種子平均,差 = 加旗標 − 正式部署,以全景片 bootstrap)", "",
         "| 標註 | 版本 | 轉介 | 自動回答中正確 | 靜默錯誤 | 異常側被轉介 / 29 | 正常牙被轉介 / 102 |", "| --- | --- | --- | --- | --- | --- | --- |"]
out = []
for lab_name, lab in labs.items():
    per = {}
    for n, rows in runs.items():
        d = defaultdict(lambda: defaultdict(list))
        for r in rows:
            a = r["alias"]
            if a not in grp:
                continue
            refer = r["refer"] == "1"
            ok = (not refer) and grp[a] == "正常牙" and r["pred"] == lab[a]["position"]
            d[a]["refer"].append(refer)
            d[a]["silent"].append((not refer) and not ok)
            d[a]["ok"].append(ok)
        per[n] = {a: {k: float(np.mean(v)) for k, v in x.items()} for a, x in d.items()}
        m = lambda k, sel=labeled: float(np.mean([per[n][a][k] for a in sel]))
        ab = [a for a in labeled if grp[a] != "正常牙"]
        nm = [a for a in labeled if grp[a] == "正常牙"]
        acc = m("ok") / (1 - m("refer"))
        lines.append(f"| {lab_name} | {'正式部署' if n == 'score_t32_640' else '+ 翻轉一致性旗標'} | {m('refer'):.1%} | {acc:.1%} | {m('silent'):.1%} | {m('refer', ab) * len(ab):.1f} | {m('refer', nm) * len(nm):.1f} |")
        out.append({"labels": lab_name, "run": n, "refer": m("refer"), "auto_acc": acc, "silent": m("silent")})
    for k, name in (("refer", "轉介"), ("silent", "靜默錯誤")):
        pp = defaultdict(lambda: [0.0, 0])
        for a in labeled:
            pp[pano[a]][0] += per["score_flipflag"][a][k] - per["score_t32_640"][a][k]
            pp[pano[a]][1] += 1
        lo, hi, p = boot(list(pp.values()))
        dlt = float(np.mean([per["score_flipflag"][a][k] - per["score_t32_640"][a][k] for a in labeled]))
        lines.append(f"| {lab_name} | 差: {name} | {100 * dlt:+.2f} 點 [{100 * lo:+.2f}, {100 * hi:+.2f}],p {p:.3f} | | | | |")
        out.append({"labels": lab_name, "run": f"diff_{k}", "diff": dlt, "lo": float(lo), "hi": float(hi), "p": float(p)})
    extra = [a for a in labeled if per["score_flipflag"][a]["refer"] > per["score_t32_640"][a]["refer"]]
    wrong = sum((per["score_t32_640"][a]["silent"] - per["score_flipflag"][a]["silent"]) for a in extra)
    more = sum((per["score_flipflag"][a]["refer"] - per["score_t32_640"][a]["refer"]) for a in extra)
    lines.append(f"| {lab_name} | 多轉介的側 | 平均 {more:.1f} 側,其中原本是靜默錯誤 {wrong:.1f} 側({wrong / more:.0%};正式部署自動回答的錯誤率 {1 - out[-4]['auto_acc']:.0%}) | | | | |")
text = "\n".join(lines)
(BASE / "score_flipflag" / "system_compare.md").write_text(text + "\n", encoding="utf-8")
with open(BASE / "score_flipflag" / "system_compare.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in out for k in r)))
    w.writeheader()
    w.writerows(out)
print(text)
