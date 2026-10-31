import argparse
import csv
import json
import pickle
import sys
from collections import Counter

import cv2
import numpy as np

from stage2.compare_yolo_unet import ROOT
from stage1.fdi_cropped import assign_fdi_indices
from stage1.fdi_numbering import assign_fdi_v2
from analysis.tufts_eval import TUFTS, universal_to_fdi

# 用 Tufts 的逐顆牙齒 GT 開發 FDI 編號(2026-10-28,CPU;快取需要 GPU 一次):
#   Tufts 上目前的編號 x4–x6 只有 69.6% 正確,錯誤幾乎都是「少一號」(tufts_eval.py)。
#   1. 快取(--build-cache): 第一階段牙齒模型(raw、imgsz 1024)預測的下顎牙齒(與 fdi_v2_check.py 相同的欄位)+ GT 下顎恆牙多邊形(FDI)。
#   2. 錯誤原因(目前的編號): 對每個被編成 x4–x6 但編錯的牙齒分類
#        midline     預測的象限與 GT 不同(中線找錯)
#        gt_missing  GT 在這顆牙與中線之間少了牙(真的缺牙: 依序編號不知道有空位)
#        not_detected GT 那些牙都在,但模型沒偵測到其中幾顆
#        other       其他(多偵測、編號多一號等)
#   3. 編號版本比較: 目前版本 vs fdi_numbering.assign_fdi_v2(v2 = 只有中線、gap_<比例> = 空隙判斷)vs molar_<門檻>(第一大臼齒錨點,見 molar_anchor)。
#      Tufts 依編號單雙分成開發組(偶數)與測試組(奇數): 只在開發組選版本,最後在測試組報告。只用恆牙的 918 張。
#      指標: 預測為 x4–x6 的牙齒編號正確率(與 tufts_eval.py 相同)、每側 x4/x5/x6 三顆都正確的比例、每側有沒有 x4–x6。
#   2026-10-28: GT 多邊形改用 tufts_teeth.merge 合併碎片(一顆牙一個輪廓),與 tufts_teeth.py 的評估相同;--fragments 用原本的碎片。
#   輸出 compare_results/tufts/fdi_variants.csv、fdi_errors.csv。
CACHE = ROOT / "compare_results" / "tufts" / "lower_teeth.pkl"
OUT = ROOT / "compare_results" / "tufts"
TARGETS = ("4", "5", "6")


def build_cache():
    from ultralytics import YOLO
    from stage1.fdi_cropped import process_and_separate_jaws
    from deployment.pipeline_timing import TEETH_MODEL
    from stage1.stage1_eval import Sized
    model = Sized(YOLO(TEETH_MODEL), 1024)
    entries = {e["External ID"].rsplit(".", 1)[0]: e for e in json.loads((TUFTS / "teeth_polygon.json").read_text())}
    cache = {}
    for path in sorted((TUFTS / "radiographs").glob("*"), key=lambda p: int(p.stem)):
        img = cv2.imread(str(path))
        _, lower = process_and_separate_jaws(img, model)
        teeth = [{k: t[k] for k in ("cx", "cy", "bbox", "area", "y_max", "score")} for t in lower]
        gt, primary = {}, False
        for o in entries[path.stem]["Label"]["objects"]:
            if not o["title"].isdigit():
                primary = primary or len(o["title"]) == 1
                continue
            fdi = universal_to_fdi(o["title"])
            if fdi and o.get("polygons") not in (None, "none"):
                polys = json.loads(o["polygons"]) if isinstance(o["polygons"], str) else o["polygons"]
                gt[fdi] = [np.array(p, dtype=np.float32) for p in polys if len(p) >= 3]
        cache[path.stem] = {"width": img.shape[1], "lower": teeth, "gt": gt, "primary": primary}
    CACHE.write_bytes(pickle.dumps(cache))
    return cache


def gt_of(point, gt):
    for fdi, polys in gt.items():
        if any(cv2.pointPolygonTest(p, point, False) >= 0 for p in polys):
            return fdi
    return None


def molar_anchor(teeth, width, ratio):
    # 以第一大臼齒為錨點(2026-10-28): 中線與象限用 v2;每側由中線往遠心,第一顆「寬度 ≥ ratio × 全景片寬」的牙齒編為 x6,
    #   其餘依相對位置編號(近心側依序 5、4、3…,遠心側 7、8)。大臼齒的寬度約是小臼齒的兩倍(Tufts GT: 中位數 0.068 vs 0.031–0.037),
    #   所以前牙缺牙或沒偵測到不會影響 x4–x6。x6 本身缺牙時第一顆大臼齒其實是 x7(會編錯 x6,但 x4、x5 仍對)。該側沒有大臼齒時沿用 v2 的編號。
    numbered = assign_fdi_v2(teeth, width)
    for quad, sign in ((4, -1), (3, 1)):
        side = sorted([t for t in numbered if t["quadrant"] == quad], key=lambda t: sign * t["cx"])
        first = next((i for i, t in enumerate(side) if (t["bbox"][2] - t["bbox"][0]) >= ratio * width), None)
        if first is None:
            continue
        for i, t in enumerate(side):
            t.pop("fdi_id", None)
            n = 6 + (i - first)
            if 1 <= n <= 8:
                t["fdi_id"] = f"{quad}{n}"
    return numbered


def number(lower, width, variant):
    teeth = [dict(t) for t in sorted(lower, key=lambda t: t["cx"])]
    if not teeth:
        return []
    if variant == "current":
        return assign_fdi_indices(teeth)
    if variant.startswith("molar_"):
        return molar_anchor(teeth, width, float(variant[6:]))  # "molar_0.05"
    return assign_fdi_v2(teeth, width, gap_ratio=None if variant == "v2" else float(variant[4:]))  # "gap_0.6"


def evaluate(cache, panos, variant, errors=None):
    teeth_n = teeth_ok = sides_n = sides_all = sides_has = 0
    for pano in panos:
        c = cache[pano]
        numbered = number(c["lower"], c["width"], variant)
        for quad in "43":
            sides_n += 1
            pred = {t["fdi_id"]: t for t in numbered if t.get("fdi_id", "")[:1] == quad and t["fdi_id"][1:] in TARGETS}
            sides_has += bool(pred)
            ok = 0
            for fdi, t in pred.items():
                g = gt_of((float(t["cx"]), float(t["cy"])), c["gt"])
                teeth_n += 1
                ok += g == fdi
                if errors is not None and g != fdi:
                    errors.append({"pano": pano, "pred": fdi, "gt": g or "", "cause": cause(fdi, g, c, numbered)})
            teeth_ok += ok
            gt_targets = [quad + n for n in TARGETS if quad + n in c["gt"]]
            sides_all += bool(gt_targets) and ok == len(pred) and set(pred) == set(gt_targets)
    return {"variant": variant, "panos": len(panos), "teeth": teeth_n, "teeth_correct": round(teeth_ok / max(teeth_n, 1), 4),
            "sides": sides_n, "side_has_target": round(sides_has / sides_n, 4), "side_all_correct": round(sides_all / sides_n, 4)}


def cause(pred, gt, c, numbered):
    if gt is None:
        return "not_on_gt_tooth"
    if gt[0] != pred[0]:
        return "midline"
    quad, k = gt[0], int(gt[1])
    missing_gt = [n for n in range(1, k) if f"{quad}{n}" not in c["gt"]]
    if missing_gt:
        return "gt_missing"
    # GT 1..k−1 都在: 看預測的同側牙齒有沒有涵蓋它們
    covered = {gt_of((float(t["cx"]), float(t["cy"])), c["gt"]) for t in numbered if t.get("quadrant") == int(quad)}
    if any(f"{quad}{n}" not in covered for n in range(1, k)):
        return "not_detected"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--gaps", nargs="+", default=["0.5", "1.0"])
    parser.add_argument("--molars", nargs="+", default=["0.045", "0.05", "0.055"], help="大臼齒的寬度門檻(全景片寬的比例)")
    parser.add_argument("--fragments", action="store_true", help="GT 用原本的碎片多邊形(2026-10-28 之前的結果)")
    parser.add_argument("--final", default=None, help="在測試組報告的版本(例如 gap_0.6);不給則只看開發組")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    cache = build_cache() if args.build_cache or not CACHE.exists() else pickle.loads(CACHE.read_bytes())
    if not args.fragments:
        from stage1.tufts_teeth import merge
        for c in cache.values():
            c["gt"] = {k: [m] for k, v in c["gt"].items() if (m := merge(v)) is not None}
    panos = [p for p, c in cache.items() if not c["primary"]]
    dev = [p for p in panos if int(p) % 2 == 0]
    test = [p for p in panos if int(p) % 2 == 1]
    print(f"恆牙 {len(panos)} 張: 開發組 {len(dev)}、測試組 {len(test)}")

    errors = []
    rows = [{"set": "dev", **evaluate(cache, dev, "current", errors)}]
    print("目前版本(開發組)錯誤原因:", Counter(e["cause"] for e in errors).most_common())
    for variant in ["v2"] + [f"gap_{g}" for g in args.gaps] + [f"molar_{r}" for r in args.molars]:
        rows.append({"set": "dev", **evaluate(cache, dev, variant)})
    if args.final:
        rows += [{"set": "test", **evaluate(cache, test, "current")}, {"set": "test", **evaluate(cache, test, args.final)}]
    for r in rows:
        print(r)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in (("fdi_variants.csv", rows), ("fdi_errors.csv", errors)):
        with open(OUT / name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)


if __name__ == "__main__":
    main()
