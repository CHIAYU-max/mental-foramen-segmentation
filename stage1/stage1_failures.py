import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

from stage2.compare_yolo_unet import ROOT
from stage1.fdi_cropped import assign_fdi_indices, draw_fdi_overlay, get_split_roi_bounds, process_and_separate_jaws
from deployment.pipeline_timing import TEETH_MODEL
from stage1.stage1_eval import Sized, preprocess

# 第一階段的失敗案例(2026-10-25)。stage1_eval 只能評估「當初有裁出 ROI」的全景片(有 ROI 才有頦孔標註,選樣偏差),
#   這裡改看 20260826__mf_851 中沒有裁出 ROI 的全景片 / 側: 重現失敗、分類原因,並看其他前處理 / imgsz 能不能救回。
#   不需要 GT: 只看有沒有裁出 ROI;救回的 ROI 是否真的包含頦孔要再目視確認(overlays/)。
# 側的定義與 fdi_cropped.get_split_roi_bounds 相同: Right = 象限 4(影像左側)、Left = 象限 3。
#   某側要有 x4 或 x5(小臼齒)才會裁 ROI。
# 原因(以基準 raw@1024 判定):
#   no_lower_jaw       沒有分出下顎牙齒(偵測不到牙齒,或過濾 / 分群後少於 2 顆)
#   no_premolar        該側沒有編到 x4、x5(缺牙,或 FDI 編號偏移)
#   reproduced_ok      重跑後其實有 ROI(與當初的結果不一致)
PANO_DIR = ROOT / "dental_project" / "20260826__mf_851"
ROI_DIR = PANO_DIR / "output" / "roi_cropped"
OUT = ROOT / "compare_results" / "stage1_failures"
CONDITIONS = [("raw", 1024), ("raw", 768), ("raw", 1280), ("clahe1", 1024), ("clahe2", 1024)]
SIDES = {"Right": 4, "Left": 3}
OVERLAY_WIDTH = 1400
# ROI 檔名是全景片檔名去掉 "0827_" 前綴(例如 0827_1000.jpg → 1000_Left_roi.jpg)
PREFIX = "0827_"


def run(img, model):
    _, lower = process_and_separate_jaws(img, model)
    lower = assign_fdi_indices(lower) if lower else []
    return lower, set(get_split_roi_bounds(lower)) if lower else set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    (OUT / "overlays").mkdir(parents=True, exist_ok=True)
    produced = Counter()
    for p in ROI_DIR.iterdir():
        for side in SIDES:
            if p.name.endswith(f"_{side}_roi.jpg"):
                produced[(p.name[:-len(f"_{side}_roi.jpg")], side)] = 1
    panos = sorted(p for p in PANO_DIR.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    roi_stem = lambda p: p.stem[len(PREFIX):] if p.stem.startswith(PREFIX) else p.stem
    failed = [(p, side) for p in panos for side in SIDES if not produced[(roi_stem(p), side)]]
    targets = sorted({p for p, _ in failed})[:args.limit or None]
    print(f"全景片 {len(panos)} 張;沒有 ROI 的側 {len(failed)} 個,分布在 {len({p for p, _ in failed})} 張全景片"
          f"(兩側都沒有 {sum(1 for p in panos if not produced[(roi_stem(p), 'Right')] and not produced[(roi_stem(p), 'Left')])} 張)")
    yolo = YOLO(TEETH_MODEL)
    rows = []
    for i, path in enumerate(targets, 1):
        raw = cv2.imread(str(path))
        results = {}
        for preproc, imgsz in CONDITIONS:
            img = preprocess(raw, preproc)
            results[(preproc, imgsz)] = run(img, Sized(yolo, imgsz))
            if (preproc, imgsz) == ("raw", 1024):
                vis = draw_fdi_overlay(raw, results[(preproc, imgsz)][0])
                h, w = vis.shape[:2]
                cv2.imwrite(str(OUT / "overlays" / f"{path.stem}.jpg"), cv2.resize(vis, (OVERLAY_WIDTH, round(h * OVERLAY_WIDTH / w))), [cv2.IMWRITE_JPEG_QUALITY, 85])
        lower, sides = results[("raw", 1024)]
        for side, q in SIDES.items():
            if produced[(roi_stem(path), side)]:
                continue
            ids = sorted(int(t["fdi_id"][1]) for t in lower if t.get("quadrant") == q)
            if side in sides:
                reason = "reproduced_ok"
            elif not lower:
                reason = "no_lower_jaw"
            else:
                reason = "no_premolar"
            rows.append({"pano": path.name, "side": side, "reason": reason, "lower_teeth": len(lower),
                         "side_teeth": len(ids), "side_fdi": " ".join(map(str, ids)),
                         **{f"{pre}_{sz}": int(side in results[(pre, sz)][1]) for pre, sz in CONDITIONS}})
        if i % 10 == 0 or i == len(targets):
            print(f"[{i}/{len(targets)}] {path.name}")
    with open(OUT / "failures.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n失敗的側 {len(rows)} 個,原因:", dict(Counter(r["reason"] for r in rows)))
    print("各條件救回(有 ROI)的側數:", {f"{pre}@{sz}": sum(r[f"{pre}_{sz}"] for r in rows) for pre, sz in CONDITIONS})
    print("該側編到的牙齒數:", dict(sorted(Counter(r["side_teeth"] for r in rows if r["reason"] == "no_premolar").items())))


if __name__ == "__main__":
    main()
