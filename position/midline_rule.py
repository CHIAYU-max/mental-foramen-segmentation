# 2026-11-05 試算「中央中線」牙位規則(使用者提議;只讀,結果印在畫面上)
# 現行(中央一半): 質心離某條牙軸 <= 0.5 r 才算該牙下方,其餘都算兩牙之間
# 中線規則: 兩條相鄰牙軸的中線 ± frac*r 內才算兩牙之間,其餘歸給比較近的那顆牙
import csv, json, sys
from collections import Counter
from pathlib import Path
import cv2
import numpy as np

ROOT = Path.home() / "YOLOv8"
sys.path.insert(0, str(ROOT))
from stage2.kfold_pano import ROI_SOURCES
from position.mf_position import ITEMS_JSON, KEY_CSV, OUT, centroid, classify
import position.mf_position_eval as ev


def half_width(poly, n):
    m = centroid(poly)
    return float(np.ptp((poly - m) @ n)) / 2


def classify_mid(poly, lines, frac=0.5):
    if any(k not in lines for k in "456"):
        return None
    m = centroid(poly)
    d = {k: float((m - c) @ n) for k, (c, _, n) in lines.items()}
    r = {k: half_width(poly, n) for k, (_, _, n) in lines.items()}
    gap45 = d["4"] - d["5"]  # 質心深度處 x4 到 x5 的距離(> 0)
    # 近心側: 用 x4–x5 間距鏡射估計 x3,x3–x4 中線在 x4 近心 gap45/2
    if d["4"] < 0:
        mid = d["4"] + gap45 / 2  # 到 x3–x4 中線的有號距離
        if abs(mid) <= frac * r["4"]:
            return 1  # 落在中線寬容範圍: 算 x4 近心(兩牙之間)
        return 2 if mid > 0 else 1
    for a, b, under_a, between, under_b in (("4", "5", 2, 3, 4), ("5", "6", 4, 5, 6)):
        if d[b] <= 0:  # 在 a 與 b 之間
            mid = (d[a] + d[b]) / 2  # > 0 = 偏 b
            rr = (r[a] + r[b]) / 2
            if abs(mid) <= frac * rr:
                return between
            return under_b if mid > 0 else under_a
    return 6


def main():
    items = json.loads(ITEMS_JSON.read_text(encoding="utf-8"))["items"]
    keys = list(csv.DictReader(open(KEY_CSV, newline="")))
    sources = {r["stem"]: r for r in csv.DictReader(open(ROI_SOURCES, newline=""))}
    folds = {r["stem"]: int(r["fold"]) for r in csv.DictReader(open(ROOT / ev.MASKS / "folds.csv", newline=""))}
    orig = {r["alias"]: r["position"] for r in csv.DictReader(open(OUT / "labels_rater.csv", newline="", encoding="utf-8-sig"))}
    adj_path = ROOT / "compare_results/mf_position/labels_rater_adjudicated.csv"
    adj = {r["alias"]: r["position"] for r in csv.DictReader(open(adj_path, newline="", encoding="utf-8-sig"))}
    fracs = [0.25, 0.5, 1.0]
    rows = []
    for k in keys:
        item = items[k["alias"]]
        cx, cy = int(k["crop_x"]), int(k["crop_y"])
        gt = np.array(item["mf"], dtype=float) + (cx, cy)
        lines = ev.lines_from_item(item, cx, cy, "a")
        cur, info = classify(gt, lines, "center")
        row = {"alias": k["alias"], "set": k["set"], "key": k["auto_position_center"], "cur": cur,
               "orig": orig.get(k["alias"], ""), "adj": adj.get(k["alias"], orig.get(k["alias"], ""))}
        for f in fracs:
            row[f"mid{f}"] = classify_mid(gt, lines, f)
        if all(x in lines for x in "45"):
            m = centroid(gt)
            d4, d5 = ((m - lines[x][0]) @ lines[x][2] for x in "45")
            row["gap45_over_r"] = (d4 - d5) / half_width(gt, lines["5"][2])
        found, poly = ev.predicted_mf(ROOT / ev.MASKS, "r34_imagenet_aug", k["stem"], "test" if k["set"] == "test" else "cv", folds.get(k["stem"]))
        if found and poly is not None:
            src = sources[k["stem"]]
            poly = poly + (int(src["x"]), int(src["y"]))
            row["p_cur"] = classify(poly, lines, "center")[0]
            row["p_mid0.5"] = classify_mid(poly, lines, 0.5)
        else:
            row["p_cur"] = row["p_mid0.5"] = None
        rows.append(row)

    print("核對: 重算現行規則 == key.csv", sum(str(r["cur"]) == r["key"] for r in rows), "/", len(rows))
    print("全部 GT 頦孔分布")
    for name in ["cur"] + [f"mid{f}" for f in fracs]:
        print(f"  {name:8s}", dict(sorted(Counter(str(r[name]) for r in rows).items())))
    g = [r["gap45_over_r"] for r in rows if "gap45_over_r" in r]
    print("x4–x5 間距 / 頦孔半寬: 中位數 %.2f, P10 %.2f, P90 %.2f" % tuple(np.percentile(g, [50, 10, 90])))
    for lab in ("orig", "adj"):
        use = [r for r in rows if r["set"] == "test" and r[lab] in tuple("123456")]
        print(f"\n測試集 vs {'原始' if lab == 'orig' else '複查後'}標註 (n={len(use)}),標註分布", dict(sorted(Counter(r[lab] for r in use).items())))
        for name in ["cur"] + [f"mid{f}" for f in fracs]:
            ok = sum(str(r[name]) == r[lab] for r in use)
            wrong = Counter((r[lab], str(r[name])) for r in use if str(r[name]) != r[lab])
            print(f"  規則 {name:8s} {ok}/{len(use)} = {ok/len(use):.3f}  錯(標註,規則): {dict(wrong.most_common(8))}")
        for name in ("p_cur", "p_mid0.5"):
            ok = sum(str(r[name]) == r[lab] for r in use)
            print(f"  U-Net 端到端 {name:9s} {ok}/{len(use)} = {ok/len(use):.3f}")
    ch = Counter((str(r["cur"]), str(r["mid0.5"])) for r in rows if str(r["cur"]) != str(r["mid0.5"]))
    print("\n現行 → 中線(0.5) 判定改變:", sum(ch.values()), "/", len(rows), dict(ch.most_common()))



if __name__ == "__main__":
    main()
