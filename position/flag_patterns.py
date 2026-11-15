"""2026-11-14 「兩種編號不一致」旗標的誤報與漏報長什麼樣子(使用者要求;CPU,只讀 score_official/per_side.csv 與標註)。"""
import csv
from collections import Counter, defaultdict

from stage2.compare_yolo_unet import ROOT
from position.position_referral import group

BASE = ROOT / "compare_results" / "deploy_eval" / "score_official"
MP = ROOT / "compare_results" / "mf_position"
adj = {r["alias"]: r for r in csv.DictReader(open(MP / "labels_adjudicated_dir/labels_rater.csv", newline="", encoding="utf-8-sig"))}
rows = [r for r in csv.DictReader(open(BASE / "per_side.csv", newline="", encoding="utf-8")) if r["seed"] == "0"]
rows = [r for r in rows if r["alias"] in adj and (adj[r["alias"]].get("position") or adj[r["alias"]].get("fdi_wrong") == "1")]
grp = {r["alias"]: group(adj[r["alias"]]) for r in rows}
lines = ["# 兩種編號不一致旗標: 狀態型態(flag_patterns.py;測試集,第一階段與種子無關)", "",
         "狀態: same 同一顆;shift±k 32 類模型的 x_k 落在現行編號差 k 號的牙;outside 不在現行任何牙的輪廓內;new_missing 只有現行有;cur_missing 只有 32 類有;both_missing 都沒有", ""]
pat = defaultdict(Counter)
per_tooth = defaultdict(Counter)
for r in rows:
    g = "正常牙" if grp[r["alias"]] == "正常牙" else grp[r["alias"]]
    st = tuple(r[f"x{k}"] for k in "456")
    kinds = sorted({s.rstrip("+-0123456789") if s.startswith("shift") else s for s in st if s != "same"})
    pat[g]["一致" if not kinds else " + ".join(kinds)] += 1
    n_bad = sum(s != "same" for s in st)
    per_tooth[g][n_bad] += 1
for g in ("正常牙", "缺牙", "FDI 錯"):
    tot = sum(pat[g].values())
    lines += [f"## {g}({tot} 側)", "", "| x4–x6 的不一致型態 | 側數 |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in pat[g].most_common()]
    lines += ["", "- 不一致的牙數(0 / 1 / 2 / 3 顆): " + " / ".join(str(per_tooth[g][i]) for i in range(4)), ""]
# 幾個事後看的規則(探索性,不能直接採用)
def rule(name, fn):
    ab = [r for r in rows if grp[r["alias"]] != "正常牙"]
    nm = [r for r in rows if grp[r["alias"]] == "正常牙"]
    return f"| {name} | {sum(fn(r) for r in ab)} / {len(ab)} | {sum(fn(r) for r in nm)} / {len(nm)} |"
st = lambda r: [r[f"x{k}"] for k in "456"]
lines += ["## 幾種規則在測試集上的偵測數(探索性: 是看過測試集才列的,只用來找方向)", "", "| 規則 | 異常側被標記 | 正常牙誤標 |", "| --- | --- | --- |",
          rule("現行: 任一顆不是 same", lambda r: any(s != "same" for s in st(r))),
          rule("至少 2 顆不是 same", lambda r: sum(s != "same" for s in st(r)) >= 2),
          rule("有 shift(整顆對到別顆牙)", lambda r: any(s.startswith("shift") for s in st(r))),
          rule("有 shift 或 cur_missing / both_missing", lambda r: any(s.startswith("shift") or s in ("cur_missing", "both_missing") for s in st(r))),
          rule("只有 outside / new_missing(沒有 shift)", lambda r: any(s != "same" for s in st(r)) and not any(s.startswith("shift") for s in st(r))),
          rule("左右顆數旗標", lambda r: r["flag_side"] == "1")]
text = "\n".join(lines)
(BASE / "flag_patterns.md").write_text(text + "\n", encoding="utf-8")
print(text)
