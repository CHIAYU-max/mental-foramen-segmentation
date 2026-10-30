import argparse
import csv
import json
import statistics
from collections import Counter

from position.mf_position import ITEMS_JSON, ROOT

OUT_DIR = ROOT / "compare_results" / "mf_position"

# 2026-10-27 缺牙候選清單(使用者: 缺牙的影像先排除,先在正常牙列上確定方法):
#   標註工具已略過「模型缺 x4、x5、x6 任一條參考線」的 19 個頦孔;這裡再用兩個規則找模型沒發現的缺牙,
#   讓標註時可以先跳過,最後仍由人確認(按 M)。
#   1. FDI 跳號: x3–x7 之間少了一個編號(例如有 34、36 沒有 35)
#   2. 空隙: 相鄰兩顆牙(依 FDI 排序)外框之間的水平距離 > --gap 倍的牙寬中位數
#      (缺一顆牙但模型把後面的牙往前編號時,編號不會跳,只會留下空隙)
#   輸出: compare_results/mf_position/missing_candidates.csv(每個頦孔一列,reason 空白表示沒有疑慮)


def x_extent(poly):
    xs = [p[0] for p in poly]
    return min(xs), max(xs)


def check(item, gap_ratio):
    q = str(item["quadrant"])
    teeth = sorted((t for t in item["teeth"] if t["fdi"][0] == q and 3 <= int(t["fdi"][1]) <= 7), key=lambda t: int(t["fdi"][1]))
    reasons, worst = [], 0.0
    nums = [int(t["fdi"][1]) for t in teeth]
    missing = [n for n in range(3, max(nums, default=3) + 1) if n not in nums and n <= 6]
    if missing:
        reasons.append("fdi_skip:" + "/".join(f"{q}{n}" for n in missing))
    widths = [x_extent(t["poly"])[1] - x_extent(t["poly"])[0] for t in teeth]
    if len(teeth) >= 2:
        w = statistics.median(widths)
        for a, b in zip(teeth, teeth[1:]):
            (a0, a1), (b0, b1) = x_extent(a["poly"]), x_extent(b["poly"])
            gap = max(b0 - a1, a0 - b1)   # 左右側方向相反,取正的那一邊
            worst = max(worst, gap / w)
            if gap > gap_ratio * w:
                reasons.append(f"gap:{a['fdi']}-{b['fdi']}")
    return reasons, worst, len(teeth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap", type=float, default=0.5, help="空隙超過牙寬中位數的幾倍算可疑")
    args = parser.parse_args()
    items = json.loads(ITEMS_JSON.read_text())["items"]
    with open(ITEMS_JSON.parent / "key.csv", newline="") as f:
        key = {r["alias"]: r for r in csv.DictReader(f)}

    rows = []
    for alias, item in items.items():
        reasons, worst, n = check(item, args.gap)
        skipped = key[alias]["reason"] != ""
        rows.append({"alias": alias, "set": key[alias]["set"], "stem": key[alias]["stem"], "teeth_x3_x7": n,
                     "max_gap_ratio": round(worst, 2), "already_skipped": int(skipped), "reason": ";".join(reasons)})

    out = OUT_DIR / "missing_candidates.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for s in ("test", "train"):
        sub = [r for r in rows if r["set"] == s]
        new = [r for r in sub if r["reason"] and not r["already_skipped"]]
        kinds = Counter(x.split(":")[0] for r in new for x in r["reason"].split(";"))
        print(f"{s}: {len(sub)} 個頦孔,工具已略過 {sum(r['already_skipped'] for r in sub)},新的缺牙候選 {len(new)} ({dict(kinds)})")
    print(f"輸出: {out}")


if __name__ == "__main__":
    main()
