"""2026-11-14 牙位標註可信度的兩份報告(使用者要求;CPU,只讀標註與已存結果)

  python -m position.position_reports midline   牙位定義的疊圖判讀結果(midline_review_server.py 寫回的 overlays.csv)
  python -m position.position_reports blind     關掉參考線重標測試集(position_server.py --blind)與原本標註、規則、自動流程的一致性

事先定的決定規則:
  midline: 只看兩種規則答案不同的 24 張。中線規則被判對的張數 vs 現行規則被判對的張數,雙尾符號檢定(不含「兩個都不對」)。
           p < 0.05 且中線較多 → 建議改用中線規則(之後要事先固定門檻、用盲標重標測試集、牙位數字全部重算);
           p < 0.05 且現行較多 → 維持現行,論文可寫「標註者在疊圖上較常同意現行規則」;否則 → 維持現行,並在 Limitations 註明兩種定義無法由現有資料區分。
  blind:   盲標(看不到牙齒輪廓、編號、參考線)與原本標註(看得到參考線)的一致率與 kappa = 同一人相隔約三週的一致性,也量出參考線對標註的影響;
           主要數字: 規則套在 GT 頦孔上的牙位 與 盲標 的一致率(原本 95.1% / 複查後 100% 有循環論證的疑慮,這個數字沒有),
           以及自動流程(5 種子)對盲標的正確率。只報告,不依結果改規則或門檻。
輸出: compare_results/mf_position/midline/review_report.md、compare_results/mf_position/blind_relabel/report.md
"""
import csv
import math
import sys
from collections import Counter, defaultdict

import numpy as np

from stage2.compare_yolo_unet import ROOT
from position.mf_position import KEY_CSV, kappas

MP = ROOT / "compare_results" / "mf_position"
CALL = "your_call(current/midline/neither)"
BLIND = ROOT / "mf_position" / "blind_relabel" / "labels_rater.csv"


def sign_p(a, b):
    n, k = a + b, min(a, b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def wilson(k, n, z=1.96):
    if n == 0:
        return "—"
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{k} / {n} = {100 * p:.1f}%({100 * (c - h):.1f}–{100 * (c + h):.1f})"


def midline():
    rows = list(csv.DictReader(open(MP / "midline" / "overlays" / "overlays.csv", newline="", encoding="utf-8-sig")))
    done = [r for r in rows if r[CALL]]
    lines = ["# 牙位定義: 疊圖判讀結果(position_reports.py midline)", "", f"- 已判 {len(done)} / {len(rows)} 張。"]
    if len(done) < len(rows):
        lines.append("- **還沒判完,以下只是目前的統計,判完再看結論。**")
    diff = [r for r in done if r["current"] != r["midline"]]
    same = [r for r in done if r["current"] == r["midline"]]
    c = Counter(r[CALL] for r in diff)
    p = sign_p(c["current"], c["midline"])
    lines += [f"- 兩種規則答案不同的 {len(diff)} 張: 現行對 {c['current']}、中線對 {c['midline']}、兩個都不對 {c['neither']};雙尾符號檢定 p = {p:.3f}",
              f"- 兩種規則答案相同的 {len(same)} 張: " + "、".join(f"{k} {v}" for k, v in Counter(r[CALL] for r in same).items()),
              "- 免標註參考(32 類牙齒遮罩)與判讀一致的張數(答案不同的張): "
              + f"根尖寬度參考 {sum(r['apex_ref'] == r[r[CALL]] for r in diff if r[CALL] in ('current', 'midline'))}、"
              + f"牙冠寬度參考 {sum(r['crown_ref'] == r[r[CALL]] for r in diff if r[CALL] in ('current', 'midline'))}(共 {c['current'] + c['midline']} 張)"]
    if len(done) == len(rows):
        if p < 0.05 and c["midline"] > c["current"]:
            verdict = "建議改用中線規則 → 下一步: 事先固定門檻、用盲標重標測試集、牙位數字全部重算(由使用者決定)"
        elif p < 0.05:
            verdict = "維持現行「中央一半」;論文可寫標註者在疊圖上較常同意現行規則"
        else:
            verdict = "維持現行「中央一半」;兩種定義無法由這 30 張區分,寫進 Limitations"
        lines.append(f"- **依事先定的規則: {verdict}**")
    text = "\n".join(lines)
    (MP / "midline" / "review_report.md").write_text(text + "\n", encoding="utf-8")
    print(text)


def blind():
    read = lambda p: {r["alias"]: r for r in csv.DictReader(open(p, newline="", encoding="utf-8-sig"))}
    if not BLIND.exists():
        print("還沒有盲標(mf_position/blind_relabel/labels_rater.csv 不存在)")
        return
    bl = {a: r["position"] for a, r in read(BLIND).items() if r["position"]}
    orig = {a: r["position"] for a, r in read(ROOT / "mf_position" / "labels_rater.csv").items()}
    adj = {a: r["position"] for a, r in read(MP / "labels_adjudicated_dir" / "labels_rater.csv").items()}
    key = {r["alias"]: r for r in csv.DictReader(open(KEY_CSV, newline="")) if r["set"] == "test"}
    total = len((ROOT / "mf_position" / "blind_relabel" / "order.txt").read_text().split())
    lines = ["# 關掉參考線重標測試集(position_reports.py blind)", "", f"- 已盲標 {len(bl)} / {total} 側。"]
    if len(bl) < total:
        lines.append("- **還沒標完,以下只是目前的統計。**")
    cat = lambda v: "正常牙" if v in tuple("123456") else {"M": "缺牙", "X": "無法判定"}.get(v, "無")
    lines.append("- 類別(盲標 × 原本): " + "、".join(f"{b} × {o} {n}" for (b, o), n in sorted(Counter((cat(bl[a]), cat(orig.get(a, ""))) for a in bl).items())))
    both = [a for a in bl if bl[a] in tuple("123456")]
    for name, ref in (("原本的標註(看得到參考線)", orig), ("複查後的標註", adj), ("規則套在 GT 頦孔上", {a: str(k["auto_position_center"]) for a, k in key.items()})):
        sub = [a for a in both if ref.get(a) in tuple("123456")]
        if sub:
            n, acc, kap, kq = kappas([bl[a] for a in sub], [ref[a] for a in sub])
            off = Counter(int(ref[a]) - int(bl[a]) for a in sub if ref[a] != bl[a])
            lines.append(f"- 盲標 vs {name}: 相同 {wilson(sum(bl[a] == ref[a] for a in sub), len(sub))},kappa {kap:.3f}(二次加權 {kq:.3f});"
                         f"不同的方向(對方 − 盲標): " + ("、".join(f"{d:+d} ×{c}" for d, c in sorted(off.items())) or "無"))
    seeds = defaultdict(dict)
    for r in csv.DictReader(open(MP / "position_seeds" / "per_seed.csv", newline="")):
        if r["variant"] == "R34/512 FP32 五折(主結果)":
            seeds[r["seed"]][r["alias"]] = r["pred"]
    for name, ref in (("盲標", bl), ("原本的標註", orig)):
        sub = [a for a in both if ref.get(a) in tuple("123456") and a in seeds["0"]]
        if sub:
            accs = [np.mean([seeds[s].get(a) == ref[a] for a in sub]) for s in sorted(seeds)]
            lines.append(f"- 自動流程(主結果 5 種子)對{name}(同一批 {len(sub)} 側): {100 * np.mean(accs):.1f} ± {100 * np.std(accs, ddof=1):.1f}%")
    text = "\n".join(lines)
    out = MP / "blind_relabel"
    out.mkdir(exist_ok=True)
    (out / "report.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    {"midline": midline, "blind": blind}[sys.argv[1]]()
