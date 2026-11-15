"""2026-11-14 翻轉一致性旗標 + 邊界距離轉介 合用的效果(使用者要求;CPU,探索性)

資料: 正式部署版本加 --flip-flag 的測試集評分(deploy_eval/score_flipflag/per_side.csv,5 種子;含 flag_flip 與 margin)。
策略(都在現行旗標之上;現行 = 沒有牙位、兩種編號不一致、或左右顆數不對稱 → 轉介):
  A  現行(正式部署)
  B  + 翻轉一致性旗標
  C  + 邊界距離最小的 10%(11/5 的 P3: 每個種子在已標註的側裡依 margin 排名取最小的 10%;要整批資料才能排名,不能逐張用)
  C' + 邊界距離 < 門檻(可逐張使用;門檻 = 開發集(CV,5 種子,單一模型)預測頦孔 margin 的第 10 百分位,不看測試集)
  D  B + C
  D' B + C'
指標: 轉介率、自動回答中正確、靜默錯誤(答錯又沒轉介;異常側沒轉介也算),對原始與複查後標註;差以全景片 bootstrap 2000 次。
10% 是 11/5 事先定的;合用是第一次在測試集上看,屬探索性。
輸出: compare_results/deploy_eval/score_flipflag/combo.md、combo.csv
"""
import csv
import json
from collections import defaultdict

import cv2
import numpy as np

import position.mf_position_eval as ev
from stage2.compare_yolo_unet import ROOT
from position.mf_position import ITEMS_JSON, KEY_CSV, centroid
from analysis.paper_metrics import read_masks
from position.position_referral import group
from position.position_seeds import boot, margin

BASE = ROOT / "compare_results" / "deploy_eval" / "score_flipflag"
MP = ROOT / "compare_results" / "mf_position"
MAIN = ROOT / ev.MASKS
TAGS = ["r34_imagenet_aug"] + [f"r34_imagenet_aug_s{s}" for s in range(1, 5)]
RATE = 0.10


def dev_threshold():
    items = json.loads(ITEMS_JSON.read_text(encoding="utf-8"))["items"]
    sources = {r["stem"]: r for r in csv.DictReader(open(ev.ROI_SOURCES, newline=""))}
    folds = {r["stem"]: int(r["fold"]) for r in csv.DictReader(open(MAIN / "folds.csv", newline=""))}
    vals = []
    for k in csv.DictReader(open(KEY_CSV, newline="")):
        if k["set"] == "test" or k["stem"] not in folds:
            continue
        lines = ev.lines_from_item(items[k["alias"]], int(k["crop_x"]), int(k["crop_y"]), "a")
        if any(q not in lines for q in "456"):
            continue
        for tag in TAGS:
            found, poly = ev.predicted_mf(MAIN, tag, k["stem"], "cv", folds[k["stem"]])
            if found and poly is not None:
                vals.append(margin(poly + (int(sources[k["stem"]]["x"]), int(sources[k["stem"]]["y"])), lines))
    return float(np.percentile(vals, 100 * RATE)), len(vals)


def main():
    thr, n_dev = dev_threshold()
    labs = {"原始標註": {r["alias"]: r for r in csv.DictReader(open(ROOT / "mf_position/labels_rater.csv", newline="", encoding="utf-8-sig"))},
            "複查後標註": {r["alias"]: r for r in csv.DictReader(open(MP / "labels_adjudicated_dir/labels_rater.csv", newline="", encoding="utf-8-sig"))}}
    adj = labs["複查後標註"]
    rows = list(csv.DictReader(open(BASE / "per_side.csv", newline="", encoding="utf-8")))
    labeled = sorted({r["alias"] for r in rows} & {a for a, r in adj.items() if r.get("position") or r.get("fdi_wrong") == "1"})
    grp = {a: group(adj[a]) for a in labeled}
    pano = {r["alias"]: r["pano"] for r in rows}
    seeds = sorted({r["seed"] for r in rows})
    by = {(r["seed"], r["alias"]): r for r in rows}
    k = int(round(RATE * len(labeled)))
    low = {}
    for s in seeds:
        mg = {a: float(by[(s, a)]["margin"]) for a in labeled if by[(s, a)]["margin"] != ""}
        low[s] = set(sorted(mg, key=mg.get)[:k])

    def refer(policy, s, a):
        r = by[(s, a)]
        out = r["pred"] == "" or r["flag_fdi"] == "1" or r["flag_side"] == "1"
        if "B" in policy:
            out |= r["flag_flip"] == "1"
        if "C" in policy:
            out |= a in low[s]
        if "T" in policy:
            out |= r["margin"] != "" and float(r["margin"]) < thr
        return out

    policies = [("A", "A 現行(正式部署)"), ("B", "B + 翻轉一致性"), ("C", "C + 邊界最近 10%(排名)"), ("T", f"C' + 邊界距離 < {thr:.3f}(開發集門檻)"),
                ("BC", "D = B + C"), ("BT", "D' = B + C'")]
    lines = ["# 翻轉一致性旗標 + 邊界距離轉介(referral_combo.py,2026-11-14,探索性)", "",
             f"- 測試集已標註 {len(labeled)} 側(正常牙 {sum(g == '正常牙' for g in grp.values())}、異常 {sum(g != '正常牙' for g in grp.values())}),5 種子平均。"
             f"邊界距離門檻 {thr:.3f} 個半寬 = 開發集(CV 預測頦孔 {n_dev} 個)的第 10 百分位;排名版每個種子取最小的 {k} 側。", ""]
    out = []
    for lab_name, lab in labs.items():
        lines += [f"## {lab_name}", "", "| 策略 | 轉介 | 自動回答中正確 | 靜默錯誤 | 異常側被轉介 / 29 | 正常牙被轉介 / 102 | 靜默錯誤: 與 A 的差 [95% CI] | 轉介: 與 A 的差 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        per = {}
        for code, name in policies:
            d = {}
            for a in labeled:
                rf = [refer(code, s, a) for s in seeds]
                ok = [(not f) and grp[a] == "正常牙" and by[(s, a)]["pred"] == lab[a]["position"] for f, s in zip(rf, seeds)]
                d[a] = {"refer": float(np.mean(rf)), "ok": float(np.mean(ok)), "silent": float(np.mean([(not f) and not o for f, o in zip(rf, ok)]))}
            per[code] = d
        m = lambda code, key, sel=labeled: float(np.mean([per[code][a][key] for a in sel]))
        ab = [a for a in labeled if grp[a] != "正常牙"]
        nm = [a for a in labeled if grp[a] == "正常牙"]

        def diff(code, ref, key):
            pp = defaultdict(lambda: [0.0, 0])
            for a in labeled:
                pp[pano[a]][0] += per[code][a][key] - per[ref][a][key]
                pp[pano[a]][1] += 1
            lo, hi, p = boot(list(pp.values()))
            return m(code, key) - m(ref, key), lo, hi, p

        for code, name in policies:
            acc = m(code, "ok") / (1 - m(code, "refer"))
            if code == "A":
                cs, cr = "—", "—"
            else:
                d, lo, hi, p = diff(code, "A", "silent")
                cs = f"{100 * d:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}],p {p:.3f}"
                cr = f"{100 * diff(code, 'A', 'refer')[0]:+.1f} 點"
                out.append({"labels": lab_name, "policy": name, "vs": "A", "silent_diff": d, "lo": lo, "hi": hi, "p": p})
            lines.append(f"| {name} | {m(code, 'refer'):.1%} | {acc:.1%} | {m(code, 'silent'):.1%} | {m(code, 'refer', ab) * len(ab):.1f} | {m(code, 'refer', nm) * len(nm):.1f} | {cs} | {cr} |")
            out.append({"labels": lab_name, "policy": name, "refer": m(code, "refer"), "auto_acc": acc, "silent": m(code, "silent"),
                        "abnormal_referred": m(code, "refer", ab) * len(ab), "normal_referred": m(code, "refer", nm) * len(nm)})
        lines.append("")
        for code, ref, text in (("BC", "C", "D 對 C(邊界排名之上再加翻轉)"), ("BC", "B", "D 對 B(翻轉之上再加邊界排名)"), ("BT", "T", "D' 對 C'"), ("BT", "B", "D' 對 B")):
            d, lo, hi, p = diff(code, ref, "silent")
            dr = diff(code, ref, "refer")[0]
            lines.append(f"- {text}: 靜默錯誤 {100 * d:+.2f} 點 [{100 * lo:+.2f}, {100 * hi:+.2f}](p {p:.3f}),轉介 {100 * dr:+.1f} 點")
            out.append({"labels": lab_name, "policy": text, "vs": ref, "silent_diff": d, "lo": lo, "hi": hi, "p": p, "refer_diff": dr})
        # 兩個旗標抓到的側重疊多少
        ov = [np.mean([(by[(s, a)]["flag_flip"] == "1") and (a in low[s]) for s in seeds]) for a in labeled]
        fl = [np.mean([by[(s, a)]["flag_flip"] == "1" for s in seeds]) for a in labeled]
        lines += [f"- 翻轉旗標成立的側平均 {sum(fl):.1f} 個,其中也在邊界最近 10% 的 {sum(ov):.1f} 個(邊界 10% 共 {k} 個)", ""]
    text = "\n".join(lines)
    (BASE / "combo.md").write_text(text + "\n", encoding="utf-8")
    with open(BASE / "combo.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(key for r in out for key in r)))
        w.writeheader()
        w.writerows(out)
    print(text)


if __name__ == "__main__":
    main()
