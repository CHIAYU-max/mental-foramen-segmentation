import argparse
import csv
import json
import statistics
from pathlib import Path

import cv2
import numpy as np

from position.mf_position import ITEMS_JSON, KEY_CSV, REPORT_DIR, classify
from position.mf_position_eval import lines_from_item

# 牙位判定邊界的敏感度(2026-10-28,不需要人工標註,CPU):
#   把 GT 頦孔輪廓沿水平方向(近遠心)平移 1 px 一步,找出最少平移多少 px 牙位就會改變(margin),
#   以 px 與頦孔等效半徑 R 報告。margin 小 = 落在判定邊界附近,頦孔預測差幾 px 或標註者判斷不同都可能改變答案。
#   另外看「預測頦孔 vs GT 頦孔牙位不同」的案例是不是集中在 margin 小的頦孔(讀 mf_position_eval.py 的輸出)。
#   參考線與規則與標註工具相同(方法 A + 中央一半 + x6 整顆牙)。
#   輸出 compare_results/mf_position/sensitivity.csv(每個頦孔一列,依 margin 由小到大),可用來決定標註時先看哪些。
MAX_SHIFT = 60


def margin(poly, lines, rule):
    base, _ = classify(poly, lines, rule)
    for s in range(1, MAX_SHIFT + 1):
        for sign in (1, -1):
            position, _ = classify(poly + (sign * s, 0), lines, rule)
            if position != base:
                return s, sign, position
    return MAX_SHIFT + 1, 0, base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule", choices=["contour", "center"], default="center")
    parser.add_argument("--models", nargs="+", default=["r34_imagenet_aug", "yolo26s-seg", "yolov8n-seg"],
                        help="讀 compare_results/mf_position/eval_<模型>_a_<規則>.csv,比較 margin 與預測改變牙位的關係")
    args = parser.parse_args()
    items = json.loads(ITEMS_JSON.read_text(encoding="utf-8"))["items"]
    with open(KEY_CSV, newline="") as f:
        keys = list(csv.DictReader(f))

    rows = []
    for k in keys:
        item = items[k["alias"]]
        crop_x, crop_y = int(k["crop_x"]), int(k["crop_y"])
        poly = np.array(item["mf"], dtype=float) + (crop_x, crop_y)
        lines = lines_from_item(item, crop_x, crop_y, "a")
        base, _ = classify(poly, lines, args.rule)
        if base is None:
            continue
        radius = (cv2.contourArea(poly.astype(np.float32)) / np.pi) ** 0.5
        s, sign, to = margin(poly, lines, args.rule)
        # 平移方向換成近遠心: 右側(第 4 象限)往 -x 是遠心
        distal = sign * (-1 if int(k["quadrant"]) == 4 else 1)
        rows.append({"alias": k["alias"], "set": k["set"], "stem": k["stem"], "position": base, "margin_px": s,
                     "margin_R": round(s / radius, 3), "radius_px": round(radius, 1),
                     "to": f"{to}({'遠心' if distal > 0 else '近心'})" if sign else ""})
    rows.sort(key=lambda r: (r["set"] != "test", r["margin_R"]))

    for model in args.models:
        path = REPORT_DIR / f"eval_{model}_a_{args.rule}.csv"
        if not path.exists():
            print(f"[略過] 沒有 {path}")
            continue
        with open(path, newline="") as f:
            ev = {r["alias"]: r for r in csv.DictReader(f)}
        for r in rows:
            e = ev.get(r["alias"])
            r[f"changed_{model}"] = "" if not e or not e["pred"] or not e["rule_gt"] else int(e["pred"] != e["rule_gt"])
            r[f"center_px_{model}"] = e["center_px"] if e else ""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "sensitivity.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(key for r in rows for key in r)))
        writer.writeheader()
        writer.writerows(rows)

    print(f"可判定的頦孔 {len(rows)} 個 → {out}")
    for subset in ("test", "train"):
        sub = [r for r in rows if r["set"] == subset]
        values = [r["margin_R"] for r in sub]
        print(f"\n{'測試集' if subset == 'test' else '訓練集'} {len(sub)} 個: margin 中位數 {statistics.median(values):.2f} R"
              f"({statistics.median(r['margin_px'] for r in sub)} px)")
        for t in (0.25, 0.5, 1.0):
            print(f"  margin ≤ {t} R: {sum(v <= t for v in values)} 個({sum(v <= t for v in values) / len(sub):.1%})")
        for model in args.models:
            key = f"changed_{model}"
            both = [r for r in sub if r.get(key, "") != ""]
            if not both:
                continue
            for lo, hi in ((0, 0.5), (0.5, 1.0), (1.0, 99)):
                group = [r for r in both if lo < r["margin_R"] <= hi or (lo == 0 and r["margin_R"] <= hi)]
                if group:
                    print(f"  {model}: margin {lo}–{hi} R 的 {len(group)} 個中,預測改變牙位 {sum(r[key] for r in group)} 個"
                          f"({sum(r[key] for r in group) / len(group):.1%})")


if __name__ == "__main__":
    main()
