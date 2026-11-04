"""2026-11-04 缺牙 / FDI 錯的偵測器: 兩種 FDI 編號是否一致(使用者決定不換 32 類模型,改用一致性當信心指標)。

- 現行編號: mf_position/items.json 的牙齒輪廓與 FDI(標註工具顯示的同一份,prepare 時的第一階段結果)。
- 32 類模型: runs/segment/tufts_fdi_all_yolov8s-seg,每類取信心最高的實例(teeth32_compare.new_teeth,imgsz 1024)。
- 比對(同 teeth32_compare.compare): 對頦孔那一側的 x4 / x5 / x6,32 類模型該編號的外框中心落在現行方法哪一顆牙的輪廓內。
- 旗標(事先固定,主要): 該側 x4–x6 任一顆不是 same → 標記「需要人看」。
  對照: 只看 x4–x5;與參考線保險(key.csv 的 axis_fallback 含 unreliable / no_root / diff)取聯集。
- 評估: 測試集的複查後標註(compare_results/mf_position/labels_rater_adjudicated.csv),
  分組 正常牙(1–6)/ 缺牙(M)/ FDI 錯;報告各組被標記的比例(異常組 = 偵測率,正常組 = 誤標率),
  以及自動流程(U-Net 預測頦孔 + 規則)在「沒被標記」的正常牙上的正確率。
- 輸出: compare_results/mf_position/fdi_flag/(per_side.csv、summary.txt)。GPU 推論 407 張全景片一次。
"""
import csv
import json
import math
import sys
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO

from stage2.compare_yolo_unet import ROOT
from position.mf_position import ITEMS_JSON, KEY_CSV
from deployment.pipeline_timing import PANO_DIR
from stage1.teeth32_compare import WEIGHTS, new_teeth

OUT = ROOT / "compare_results" / "mf_position" / "fdi_flag"
LABELS = ROOT / "compare_results" / "mf_position" / "labels_rater_adjudicated.csv"
EVAL = ROOT / "compare_results" / "mf_position" / "eval_r34_imagenet_aug_a_center.csv"
AXIS_FLAGS = ("unreliable", "no_root", "diff")


def side_status(cur, new, quad):
    # cur: [(fdi, 全景片座標輪廓)];new: {fdi: (box, conf)};回傳 {x4/x5/x6: status}
    out = {}
    cur_ids = {f for f, _ in cur}
    for n in "456":
        fdi = quad + n
        nw = new.get(fdi)
        if nw is None:
            out[n] = "new_missing" if fdi in cur_ids else "both_missing"
            continue
        if fdi not in cur_ids:
            out[n] = "cur_missing"
            continue
        x1, y1, x2, y2 = nw[0]
        center = (float((x1 + x2) / 2), float((y1 + y2) / 2))
        hit = next((f for f, c in cur if cv2.pointPolygonTest(c.astype(np.int32), center, False) >= 0), None)
        if hit is None:
            out[n] = "outside"
        elif hit == fdi:
            out[n] = "same"
        elif hit[0] == quad:
            out[n] = f"shift{int(hit[1]) - int(n):+d}"
        else:
            out[n] = "other_side"
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return "—"
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{k}/{n} = {100 * p:.1f}% ({100 * (c - h):.1f}–{100 * (c + h):.1f})"


def main():
    sys.stdout.reconfigure(line_buffering=True)
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads(ITEMS_JSON.read_text())
    with open(KEY_CSV, newline="") as f:
        key = {r["alias"]: r for r in csv.DictReader(f)}
    model = YOLO(WEIGHTS)
    cache = {}
    rows = []
    for alias in data["order"]:
        k, item = key[alias], data["items"][alias]
        pano = k["pano"]
        if pano not in cache:
            img = cv2.imread(str(PANO_DIR / pano))
            cache[pano] = new_teeth(model, img)
        off = np.array([float(k["crop_x"]), float(k["crop_y"])])
        cur = [(t["fdi"], np.asarray(t["poly"], dtype=float) + off) for t in item["teeth"] if len(t["fdi"]) == 2]
        st = side_status(cur, cache[pano], str(k["quadrant"]))
        rows.append({"alias": alias, "set": k["set"], **{f"x{n}": st[n] for n in "456"},
                     "flag_x456": int(any(st[n] != "same" for n in "456")),
                     "flag_x45": int(any(st[n] != "same" for n in "45")),
                     "flag_axis": int(any(a in k["axis_fallback"] for a in AXIS_FLAGS) or bool(k["reason"]))})
    print(f"全景片 {len(cache)} 張、頦孔 {len(rows)} 個")
    with open(OUT / "per_side.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    with open(LABELS, newline="", encoding="utf-8-sig") as f:
        lab = {r["alias"]: r for r in csv.DictReader(f)}
    with open(EVAL, newline="") as f:
        pred = {r["alias"]: r["pred"] for r in csv.DictReader(f)}
    group = lambda r: "FDI 錯" if r["fdi_wrong"] == "1" else ("缺牙" if r["position"] == "M" else ("正常牙" if r["position"] in list("123456") else None))
    test = [r for r in rows if r["set"] == "test" and r["alias"] in lab and group(lab[r["alias"]])]
    lines = [f"測試集已標註 {len(test)} 個:" + str(dict(Counter(group(lab[r['alias']]) for r in test)))]
    flags = {"兩種編號 x4–x6 不一致(主要)": lambda r: r["flag_x456"], "只看 x4–x5": lambda r: r["flag_x45"],
             "參考線保險": lambda r: r["flag_axis"], "x4–x6 不一致 或 參考線保險": lambda r: r["flag_x456"] or r["flag_axis"]}
    for name, fn in flags.items():
        lines.append(f"\n== {name}")
        for g in ("缺牙", "FDI 錯", "正常牙"):
            items = [r for r in test if group(lab[r["alias"]]) == g]
            lines.append(f"  {g:4s} 被標記 {wilson(sum(bool(fn(r)) for r in items), len(items))}")
        ab = [r for r in test if group(lab[r["alias"]]) != "正常牙"]
        lines.append(f"  異常合計 偵測率 {wilson(sum(bool(fn(r)) for r in ab), len(ab))}")
        kept = [r for r in test if not fn(r)]
        kept_normal = [r for r in kept if group(lab[r["alias"]]) == "正常牙"]
        lines.append(f"  沒被標記的 {len(kept)} 個中,異常混入 {len(kept) - len(kept_normal)} 個;"
                     f"其中正常牙的自動流程正確 {wilson(sum(pred.get(r['alias']) == lab[r['alias']]['position'] for r in kept_normal), len(kept_normal))}")
    lines.append("\n狀態分布(測試集,異常組): " + str(dict(Counter(r[f"x{n}"] for r in test if group(lab[r["alias"]]) != "正常牙" for n in "456"))))
    lines.append("狀態分布(測試集,正常牙): " + str(dict(Counter(r[f"x{n}"] for r in test if group(lab[r["alias"]]) == "正常牙" for n in "456"))))
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
