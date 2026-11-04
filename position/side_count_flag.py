"""2026-11-04 左右兩側顆數比較當缺牙旗標(探索性;GPU 推論 407 張一次)。

想法: 同一張全景片左右兩側的下排牙齒應該對稱;頦孔那一側比對側少,就可能缺牙。gap_flag.py 發現 6 個缺牙的空位已被鄰牙填掉、
幾何上看不出來,這裡看顆數差能不能抓到它們。
- 牙齒: 第一階段牙齒模型(與 mf_position prepare 相同: process_and_separate_jaws + assign_fdi_indices),每象限 x1–x7 的顆數。
  同時用 32 類模型(teeth32_compare.new_teeth)各象限 x1–x7 出現的類別數當對照。
- 特徵: diff = 對側顆數 − 頦孔側顆數(> 0 = 頦孔側較少);門檻事先固定 diff ≥ 1。
- 評估: 測試集複查後標註;與 fdi_flag(兩種編號不一致)取聯集。
- 輸出 compare_results/mf_position/side_count/(per_side.csv、summary.txt)。
"""
import csv
import math

import cv2
from ultralytics import YOLO

from stage2.compare_yolo_unet import ROOT
from stage1.fdi_cropped import assign_fdi_indices, process_and_separate_jaws
from position.mf_position import KEY_CSV
from deployment.pipeline_timing import PANO_DIR, TEETH_MODEL
from stage1.stage1_eval import Sized
from stage1.teeth32_compare import WEIGHTS, new_teeth

OUT = ROOT / "compare_results" / "mf_position" / "side_count"
LABELS = ROOT / "compare_results" / "mf_position" / "labels_rater_adjudicated.csv"
FDI_FLAG = ROOT / "compare_results" / "mf_position" / "fdi_flag" / "per_side.csv"


def wilson(k, n, z=1.96):
    if n == 0:
        return "—"
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{k}/{n} = {100 * p:.1f}% ({100 * (c - h):.1f}–{100 * (c + h):.1f})"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with open(KEY_CSV, newline="") as f:
        keys = list(csv.DictReader(f))
    teeth_model = Sized(YOLO(TEETH_MODEL), 1024)
    model32 = YOLO(WEIGHTS)
    counts = {}
    for pano in sorted({k["pano"] for k in keys}):
        img = cv2.imread(str(PANO_DIR / pano))
        _, lower = process_and_separate_jaws(img, teeth_model)
        lower = assign_fdi_indices(lower) if lower else []
        cur = {q: sum(1 for t in lower if t.get("fdi_id", "")[:1] == q and t["fdi_id"][1:] in "1234567") for q in "34"}
        new = new_teeth(model32, img)
        n32 = {q: sum(1 for f in new if f[0] == q and f[1] in "1234567") for q in "34"}
        counts[pano] = (cur, n32)
    with open(FDI_FLAG, newline="") as f:
        fdi = {r["alias"]: int(r["flag_x456"]) for r in csv.DictReader(f)}
    rows = []
    for k in keys:
        q = str(k["quadrant"])
        o = "3" if q == "4" else "4"
        cur, n32 = counts[k["pano"]]
        rows.append({"alias": k["alias"], "set": k["set"], "count": cur[q], "count_other": cur[o], "diff": cur[o] - cur[q],
                     "count32": n32[q], "count32_other": n32[o], "diff32": n32[o] - n32[q], "flag_fdi": fdi.get(k["alias"], "")})
    with open(OUT / "per_side.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    with open(LABELS, newline="", encoding="utf-8-sig") as f:
        lab = {r["alias"]: r for r in csv.DictReader(f)}
    group = lambda r: "FDI 錯" if r["fdi_wrong"] == "1" else ("缺牙" if r["position"] == "M" else ("正常牙" if r["position"] in list("123456") else None))
    test = [r for r in rows if r["set"] == "test" and r["alias"] in lab and group(lab[r["alias"]])]
    by = {g: [r for r in test if group(lab[r["alias"]]) == g] for g in ("缺牙", "FDI 錯", "正常牙")}
    rules = {"現行模型 對側多 ≥ 1 顆": lambda r: r["diff"] >= 1, "32 類模型 對側多 ≥ 1 類": lambda r: r["diff32"] >= 1,
             "兩種編號不一致(fdi_flag)": lambda r: r["flag_fdi"] == 1,
             "兩種編號不一致 或 現行模型對側多 ≥ 1": lambda r: r["flag_fdi"] == 1 or r["diff"] >= 1,
             "兩種編號不一致 或 32 類對側多 ≥ 1": lambda r: r["flag_fdi"] == 1 or r["diff32"] >= 1}
    lines = [f"測試集 {len(test)} 個: " + str({g: len(v) for g, v in by.items()})]
    for name, fn in rules.items():
        lines.append(f"\n== {name}")
        for g, v in by.items():
            lines.append(f"  {g:4s} 被標記 {wilson(sum(bool(fn(r)) for r in v), len(v))}")
        ab = by["缺牙"] + by["FDI 錯"]
        lines.append(f"  異常合計 {wilson(sum(bool(fn(r)) for r in ab), len(ab))}")
    six = ("p013R", "p017L", "p059R", "p062L", "p062R", "p077R")
    lines.append("\nfdi_flag 漏掉的 6 個缺牙(alias 頦孔側/對側 現行、32 類): " +
                 "; ".join(f"{r['alias']} {r['count']}/{r['count_other']}、{r['count32']}/{r['count32_other']}" for r in test if r["alias"] in six))
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
