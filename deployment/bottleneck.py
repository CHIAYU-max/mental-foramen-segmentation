"""2026-11-14 正式部署版本的精確度卡在哪(使用者要求;CPU,只讀 score_official 的輸出)。

把測試集每一側 × 每個種子的結果分成互斥的類別,看「沒有自動答對」的側各自卡在哪一段:
  異常側(缺牙 / FDI 錯): 被轉介(正確處理)/ 沒被轉介(靜默錯誤,第一階段的偵測器漏掉)
  正常牙: 自動答對 / 被轉介(依原因: 沒有牙位、兩種編號不一致、左右顆數、只有邊界距離;並看如果不轉介答案會不會對)/
          自動答錯(頦孔沒命中 = 第二階段;頦孔有命中但牙位錯 = 邊界 / 參考線 / 標註)
輸出: compare_results/deploy_eval/score_official/bottleneck.md、bottleneck.csv
"""
import csv
from collections import Counter, defaultdict

import numpy as np

from stage2.compare_yolo_unet import ROOT
from position.mf_position import KEY_CSV
from position.position_referral import group

BASE = ROOT / "compare_results" / "deploy_eval" / "score_official"
MP = ROOT / "compare_results" / "mf_position"


def main():
    labs = {"原始標註": {r["alias"]: r for r in csv.DictReader(open(ROOT / "mf_position/labels_rater.csv", newline="", encoding="utf-8-sig"))},
            "複查後標註": {r["alias"]: r for r in csv.DictReader(open(MP / "labels_adjudicated_dir/labels_rater.csv", newline="", encoding="utf-8-sig"))}}
    adj = labs["複查後標註"]
    key = {r["alias"]: r for r in csv.DictReader(open(KEY_CSV, newline=""))}
    vis = {r["stem"]: int(r["mf_visible"]) for r in csv.DictReader(open(ROOT / "visibility_grading" / "visibility.csv", newline=""))}
    rows = list(csv.DictReader(open(BASE / "per_side.csv", newline="", encoding="utf-8")))
    hit = {(r["seed"], r["stem"]): float(r["mf_hit_1R"]) for r in csv.DictReader(open(BASE / "per_roi.csv", newline="", encoding="utf-8"))}
    labeled = sorted({r["alias"] for r in rows} & {a for a, r in adj.items() if r.get("position") or r.get("fdi_wrong") == "1"})
    grp = {a: group(adj[a]) for a in labeled}
    seeds = sorted({r["seed"] for r in rows})
    n = len(labeled) * len(seeds)
    lines = ["# 正式部署版本的精確度卡在哪(bottleneck.py,2026-11-14;測試集已標註 "
             f"{len(labeled)} 側 × {len(seeds)} 種子,比例 = 佔全部的側)", ""]
    out = []
    for lab_name, lab in labs.items():
        c = Counter()
        would = Counter()
        visc = Counter()
        for r in rows:
            a = r["alias"]
            if a not in grp:
                continue
            refer = r["refer"] == "1"
            reason = "沒有牙位(沒有頦孔或參考線不完整)" if r["pred"] == "" else "兩種編號不一致" if r["flag_fdi"] == "1" else "左右顆數不對稱" if r["flag_side"] == "1" else "只有邊界距離"
            if grp[a] != "正常牙":
                c[("異常側", "被轉介(正確處理)" if refer else "沒被轉介 → 靜默錯誤(偵測器漏掉)")] += 1
                if refer:
                    would[("異常側", reason)] += 1
                continue
            ok = r["pred"] == lab[a]["position"]
            h = hit.get((r["seed"], key[a]["stem"]))
            if refer:
                c[("正常牙被轉介", reason)] += 1
                would[("正常牙被轉介", reason, "不轉介會答對" if ok else "不轉介會答錯")] += 1
            elif ok:
                c[("正常牙", "自動答對")] += 1
            else:
                cause = "頦孔沒命中(第二階段)" if h == 0 else "頦孔有命中、牙位仍錯(邊界 / 參考線 / 標註)"
                c[("正常牙自動答錯", cause)] += 1
                visc[("正常牙自動答錯", "可辨識" if vis.get(key[a]["stem"]) == 1 else "無法辨識")] += 1
        lines += [f"## {lab_name}", "", "| 類別 | 細項 | 佔全部的側 | 側數(5 種子平均) |", "| --- | --- | --- | --- |"]
        for (g, d), v in sorted(c.items(), key=lambda kv: (kv[0][0], -kv[1])):
            lines.append(f"| {g} | {d} | {v / n:.1%} | {v / len(seeds):.1f} |")
            out.append({"labels": lab_name, "group": g, "detail": d, "share": v / n, "sides": v / len(seeds)})
        ok_auto = c[("正常牙", "自動答對")]
        ref = sum(v for (g, _), v in c.items() if g == "正常牙被轉介") + c[("異常側", "被轉介(正確處理)")]
        silent = n - ok_auto - ref
        lines += ["", f"- 合計: 自動答對 {ok_auto / n:.1%}、轉介 {ref / n:.1%}、靜默錯誤 {silent / n:.1%}(其中異常側沒被轉介 "
                  f"{c[('異常側', '沒被轉介 → 靜默錯誤(偵測器漏掉)')] / n:.1%}、正常牙答錯 {sum(v for (g, _), v in c.items() if g == '正常牙自動答錯') / n:.1%})",
                  "- 正常牙被轉介時,如果不轉介答案會不會對: " + ";".join(f"{k[1]} {k[2]} {v / len(seeds):.1f} 側" for k, v in sorted(would.items()) if k[0] == "正常牙被轉介"),
                  "- 異常側被轉介的原因: " + ";".join(f"{k[1]} {v / len(seeds):.1f} 側" for k, v in sorted(would.items()) if k[0] == "異常側"),
                  "- 正常牙自動答錯的可辨識度: " + ";".join(f"{k[1]} {v / len(seeds):.1f} 側" for k, v in sorted(visc.items())), ""]
    text = "\n".join(lines)
    (BASE / "bottleneck.md").write_text(text + "\n", encoding="utf-8")
    with open(BASE / "bottleneck.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(text)


if __name__ == "__main__":
    main()
