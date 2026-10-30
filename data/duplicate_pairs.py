import csv
import statistics
from collections import defaultdict

import cv2
import numpy as np

from stage2.compare_preprocessing import collect_images
from stage2.compare_yolo_unet import ROOT, load_gt, pixel_metrics
from data.find_same_patient import find_pano
from stage2.kfold_patient import OUT_DIR as PATIENT_DIR
from stage2.kfold_patient import UNET_TAG

# 第一批與第二批中「同一病人同一側」的 ROI 是從同一張全景片裁出來、各自標註的。
# 把兩個 ROI 都放回全景片座標(模板比對找出裁切位置),就能直接比較:
#   1. 同一個頦孔的兩次標註有多一致(樣本數遠大於重新標註的 40 張)
#   2. 兩次裁切的範圍差多少、可辨識性判定是否一致
#   3. 交叉驗證中,重複的那一張被放在訓練集時,分數是否虛高(洩漏的實際影響)
ORIGINAL_DIR = "dental_project/20260909/mf_559_original"
MERGED_CSV = ROOT / "compare_results" / "same_patient_check" / "merged_patients.csv"
OUT_CSV = ROOT / "compare_results" / "same_patient_check" / "duplicate_pairs.csv"


def locate(pano, roi):
    # ROI 是全景片的直接裁切(未縮放),回傳 (x, y, 比對分數)
    result = cv2.matchTemplate(pano, roi, cv2.TM_CCOEFF_NORMED)
    _, score, _, (x, y) = cv2.minMaxLoc(result)
    return x, y, float(score)


def place(mask, shape, x, y):
    canvas = np.zeros(shape, dtype=bool)
    h, w = mask.shape
    canvas[y : y + h, x : x + w] = mask
    return canvas


def center(mask):
    return np.argwhere(mask).mean(0) if mask.any() else None


def main():
    with open(MERGED_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    images = collect_images((ROOT / ORIGINAL_DIR).resolve())
    with open(PATIENT_DIR / "compare_cv.csv", newline="") as f:
        model_dice = {r["filename"]: float(r["dice"]) for r in csv.DictReader(f) if r["variant"] == UNET_TAG and r["class"] == "mf"}
    visibility_csv = ROOT / "visibility_grading" / "visibility.csv"
    visible = {}
    if visibility_csv.exists():
        with open(visibility_csv, newline="") as f:
            visible = {r["stem"]: int(r["mf_visible"]) for r in csv.DictReader(f)}

    groups = defaultdict(list)
    for r in rows:
        groups[(r["merged_patient"], r["side"])].append(r)
    pairs = [g for g in groups.values() if len(g) == 2]
    print(f"同一病人同一側的 ROI 共 {len(pairs)} 組(另有 {sum(len(g) > 2 for g in groups.values())} 組超過兩張,不列入)")

    records, low_match = [], 0
    for pair in pairs:
        pair = sorted(pair, key=lambda r: r["batch"])
        pano = cv2.imread(str(find_pano(pair[0]["batch"], pair[0]["number"])), cv2.IMREAD_GRAYSCALE)
        placed, rects = [], []
        for r in pair:
            roi = cv2.imread(str(images[r["stem"]]), cv2.IMREAD_GRAYSCALE)
            x, y, score = locate(pano, roi)
            low_match += score < 0.9
            gt = load_gt(r["stem"], roi.shape, images[r["stem"]].parent.parent / "labels", {})
            placed.append({cls: place(m, pano.shape, x, y) for cls, m in gt.items()})
            rects.append((x, y, roi.shape[1], roi.shape[0], score))
        a, b = placed
        # 兩次裁切範圍的重疊程度
        (x1, y1, w1, h1, s1), (x2, y2, w2, h2, s2) = rects
        inter = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        crop_iou = inter / (w1 * h1 + w2 * h2 - inter)
        ca, cb = center(a["mf"]), center(b["mf"])
        records.append({
            "first": pair[0]["stem"], "second": pair[1]["stem"], "side": pair[0]["side"],
            "same_fold": pair[0]["fold"] == pair[1]["fold"],
            "match_first": round(s1, 3), "match_second": round(s2, 3), "crop_iou": round(crop_iou, 3),
            "mf_dice_between_labels": round(pixel_metrics(a["mf"], b["mf"])["dice"], 4),
            "ian_dice_between_labels": round(pixel_metrics(a["ian"], b["ian"])["dice"], 4),
            "mf_center_distance": round(float(np.linalg.norm(ca - cb)), 1) if ca is not None and cb is not None else "",
            "mf_area_first": int(a["mf"].sum()), "mf_area_second": int(b["mf"].sum()),
            "visible_first": visible.get(pair[0]["stem"], ""), "visible_second": visible.get(pair[1]["stem"], ""),
            "model_dice_first": model_dice.get(pair[0]["stem"], ""), "model_dice_second": model_dice.get(pair[1]["stem"], ""),
        })

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    n = len(records)
    print(f"模板比對分數低於 0.9 的 ROI: {low_match} 張(代表該 ROI 可能不是原圖的直接裁切,位置可能不準)")
    print(f"兩次裁切範圍的 IoU 中位數: {statistics.median(r['crop_iou'] for r in records):.3f}")

    print("\n===== 同一個頦孔、兩次標註的一致性(第一批 vs 第二批,全景片座標) =====")
    mf = [r["mf_dice_between_labels"] for r in records]
    dist = [r["mf_center_distance"] for r in records if r["mf_center_distance"] != ""]
    print(f"頦孔 Dice: 平均 {statistics.mean(mf):.4f}、中位數 {statistics.median(mf):.4f};Dice < 0.5 的有 {sum(d < 0.5 for d in mf)} / {n} 組")
    print(f"頦孔中心距離(像素): 中位數 {statistics.median(dist):.1f};超過 20 像素的有 {sum(d > 20 for d in dist)} 組")
    print(f"頦孔面積比 第二批/第一批: 中位數 {statistics.median(r['mf_area_second'] / r['mf_area_first'] for r in records if r['mf_area_first']):.2f}")
    print(f"神經管 Dice: 平均 {statistics.mean(r['ian_dice_between_labels'] for r in records):.4f}")

    if visible:
        print("\n===== 同一個區域,兩個版本的可辨識性判定 =====")
        both = [(r["visible_first"], r["visible_second"]) for r in records if r["visible_first"] != "" and r["visible_second"] != ""]
        for vf in (1, 0):
            for vs in (1, 0):
                print(f"  第一批 {'可辨識' if vf else '無法辨識'} / 第二批 {'可辨識' if vs else '無法辨識'}: {sum(1 for a, b in both if a == vf and b == vs)} 組")

    print("\n===== 同一個區域,模型(最終 U-Net,交叉驗證)在兩個版本上的頦孔 Dice =====")
    scored = [r for r in records if r["model_dice_first"] != "" and r["model_dice_second"] != ""]
    print(f"第一批版本 {statistics.mean(r['model_dice_first'] for r in scored):.4f}、第二批版本 {statistics.mean(r['model_dice_second'] for r in scored):.4f}(n={len(scored)})")

    print("\n===== 洩漏的影響:重複的另一張是否在訓練集中 =====")
    for label, same in (("兩張在同一折(彼此都沒被訓練過)", True), ("兩張在不同折(另一張在訓練集中)", False)):
        sub = [r for r in scored if r["same_fold"] == same]
        if sub:
            values = [r["model_dice_first"] for r in sub] + [r["model_dice_second"] for r in sub]
            print(f"  {label}: {len(sub)} 組,模型頦孔 Dice 平均 {statistics.mean(values):.4f}")
    print(f"\n逐組結果: {OUT_CSV}")


if __name__ == "__main__":
    main()
