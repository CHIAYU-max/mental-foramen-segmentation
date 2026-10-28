import argparse
import csv
import re
from collections import defaultdict

import cv2
import numpy as np

from compare_preprocessing import collect_images
from compare_yolo_unet import ROOT, index_files
from kfold_patient import OUT_DIR as PATIENT_DIR

# 不依賴檔名或編號,直接以影像內容找出每張 ROI 是從哪一張全景片、哪個位置裁出來的。
#   1. 在縮小的影像上,對每張全景片做多尺度模板比對,取分數最高的幾張候選
#   2. 在候選上以原始解析度驗證,取得精確的位置與分數
# 有了真正的來源全景片,就能可靠地判斷:同一病人(同一張全景片)、同一區域被裁兩次、測試集與訓練資料的重疊。
TRAIN_ROI_DIR = "dental_project/20260909/mf_559_original"
TEST_ROI_DIR = "dental_project/20261008/test"
PANO_DIR = "dental_project/original_pictures"
OUT_DIR = ROOT / "compare_results" / "same_patient_check"
SMALL = 0.2
SCALES = [0.8, 1.0, 1.25]
CANDIDATES = 3
ACCEPT = 0.85
SIDE = re.compile(r"_(Left|Right)_roi$")


def prepare(image, factor):
    small = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(small)


def best_match(pano, roi):
    if roi.shape[0] > pano.shape[0] or roi.shape[1] > pano.shape[1]:
        return -1.0, (0, 0)
    result = cv2.matchTemplate(pano, roi, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(result)
    return float(score), loc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="只處理前 N 張 ROI(測試用)")
    parser.add_argument("--summarize-only", action="store_true", help="不重新比對,只用已存的 roi_sources.csv 重新彙整")
    parser.add_argument("--min-score", type=float, default=0.9, help="可信比對的最低分數")
    parser.add_argument("--min-gap", type=float, default=0.06, help="可信比對須高於第二名的差距")
    args = parser.parse_args()

    if args.summarize_only:
        with open(OUT_DIR / "roi_sources.csv", newline="") as f:
            records = list(csv.DictReader(f))
        for r in records:
            for key in ("score", "rx", "ry", "rw", "rh"):
                r[key] = float(r[key])
            r["second_score"] = float(r["second_score"]) if r["second_score"] else ""
        summarize(records, args.min_score, args.min_gap)
        return

    # 全景片:original_pictures 及其子資料夾(20260826__mf_851 是從這裡挑出來的,內容相同,不重複搜尋)
    # 原始解析度的全景片只在精確驗證時才讀取,避免同時佔用大量記憶體
    panos, small_panos = {}, {}
    for path in sorted((ROOT / PANO_DIR).rglob("*")):
        if not path.is_file() or "labels" in path.parts:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            name = str(path.relative_to(ROOT / PANO_DIR))
            panos[name] = path
            small_panos[name] = prepare(image, SMALL)
    # 同一張全景片可能在不同資料夾各有一份(例如 new_pictures 與上層資料夾),先把完全相同的影像歸為同一組
    names = list(small_panos)
    prints = np.stack([cv2.resize(small_panos[n], (128, 64), interpolation=cv2.INTER_AREA).astype(np.float64).ravel() for n in names])
    prints = (prints - prints.mean(1, keepdims=True)) / (prints.std(1, keepdims=True) + 1e-6)
    corr = prints @ prints.T / prints.shape[1]
    group = {}
    for i, name in enumerate(names):
        same = [names[j] for j in np.where(corr[i] > 0.98)[0] if names[j] in group]
        group[name] = group[same[0]] if same else name
    n_groups = len(set(group.values()))
    print(f"全景片 {len(panos)} 張,其中不重複的 {n_groups} 張(其餘是同一張影像在不同資料夾的副本)")

    rois = [(s, p, "train") for s, p in sorted(collect_images((ROOT / TRAIN_ROI_DIR).resolve()).items())]
    rois += [(s, p, "test") for s, p in sorted(index_files((ROOT / TEST_ROI_DIR).resolve()).items())]
    if args.limit:
        rois = rois[: args.limit]

    with open(PATIENT_DIR / "folds.csv", newline="") as f:
        fold_of = {r["stem"]: r["fold"] for r in csv.DictReader(f)}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "roi_sources.csv"
    records = []
    for i, (stem, path, dataset) in enumerate(rois, start=1):
        roi = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        # 粗搜尋:所有全景片 × 各尺度
        coarse = []
        for name, small in small_panos.items():
            score = max(best_match(small, prepare(roi, SMALL * s))[0] for s in SCALES)
            coarse.append((score, name))
        coarse.sort(reverse=True)
        # 每組相同影像只保留一個候選,候選才會是真正不同的全景片
        distinct, seen = [], set()
        for score, name in coarse:
            if group[name] not in seen:
                seen.add(group[name])
                distinct.append((score, name))
        coarse = distinct
        # 精確驗證:原始解析度
        best = (-1.0, "", 1.0, (0, 0), (1, 1))
        for _, name in coarse[:CANDIDATES]:
            full = cv2.imread(str(panos[name]), cv2.IMREAD_GRAYSCALE)
            for s in SCALES:
                template = roi if s == 1.0 else cv2.resize(roi, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                score, loc = best_match(full, template)
                if score > best[0]:
                    best = (score, name, s, loc, full.shape)
        score, name, scale, (x, y), (pano_h, pano_w) = best
        records.append({
            "stem": stem, "set": dataset, "fold": fold_of.get(stem, ""), "side": "" if dataset == "train" else SIDE.search(stem)[1] if SIDE.search(stem) else "",
            "source_pano": name if score >= ACCEPT else "", "pano_group": group[name] if score >= ACCEPT else "",
            "score": round(score, 3), "second_score": round(coarse[1][0], 3) if len(coarse) > 1 else "",
            "scale": scale, "x": x, "y": y, "w": int(roi.shape[1] * scale), "h": int(roi.shape[0] * scale),
            # 以全景片寬高為單位的相對座標:同一張影像的不同副本解析度可能不同
            "rx": round(x / pano_w, 4), "ry": round(y / pano_h, 4),
            "rw": round(roi.shape[1] * scale / pano_w, 4), "rh": round(roi.shape[0] * scale / pano_h, 4),
        })
        if i % 25 == 0 or i == len(rois):
            print(f"[{i}/{len(rois)}] {stem}: {name} score {score:.3f}")
            with open(out, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
                writer.writeheader()
                writer.writerows(records)

    summarize(records, args.min_score, args.min_gap)


def group_of(stem, dataset):
    return "test" if dataset == "test" else stem.split("_")[0]


def summarize(records, min_score, min_gap):
    # 只採用「分數高、而且明顯高於其他全景片」的比對;分數普通且與第二名接近的,多半是來源全景片不在搜尋範圍內的誤配
    for r in records:
        second = r["second_score"] if r["second_score"] != "" else 0.0
        r["confident"] = bool(r["source_pano"]) and r["score"] >= min_score and r["score"] - second >= min_gap
    print(f"\n可信的比對: 分數 ≥ {min_score} 且比第二名高 {min_gap} 以上")
    for label in ("first", "second", "test"):
        items = [r for r in records if group_of(r["stem"], r["set"]) == label]
        print(f"  {label:<7} {sum(r['confident'] for r in items)} / {len(items)} 張找到可信的來源全景片")
    found = [r for r in records if r["confident"]]

    # 以「相同影像的組別」判斷病人,不同資料夾裡的同一張全景片視為同一位病人
    by_pano = defaultdict(list)
    for r in found:
        by_pano[r["pano_group"]].append(r)
    train_panos = {r["pano_group"] for r in found if r["set"] == "train"}
    print(f"\n可信比對的訓練 ROI 來自 {len(train_panos)} 張不同的全景片")
    test_overlap = [r for r in found if r["set"] == "test" and r["pano_group"] in train_panos]
    print(f"測試 ROI 的來源全景片也出現在訓練資料中: {len(test_overlap)} 張")
    for r in test_overlap:
        partners = [t["stem"] for t in by_pano[r["pano_group"]] if t["set"] == "train"]
        print(f"  {r['stem']}  來源 {r['source_pano']}  訓練中同一張全景片的 ROI: {partners}")

    # 同一張全景片上,裁切範圍重疊的訓練 ROI(同一區域被裁兩次)
    duplicates, split = [], []
    for name, items in by_pano.items():
        train_items = [r for r in items if r["set"] == "train"]
        if len({r["fold"] for r in train_items}) > 1:
            split.append(name)
        for a in range(len(train_items)):
            for b in range(a + 1, len(train_items)):
                p, q = train_items[a], train_items[b]
                ix = max(0, min(p["rx"] + p["rw"], q["rx"] + q["rw"]) - max(p["rx"], q["rx"]))
                iy = max(0, min(p["ry"] + p["rh"], q["ry"] + q["rh"]) - max(p["ry"], q["ry"]))
                if ix * iy > 0.3 * min(p["rw"] * p["rh"], q["rw"] * q["rh"]):
                    duplicates.append((p["stem"], q["stem"], p["fold"], q["fold"], name))
    print(f"同一張全景片、裁切範圍大幅重疊的訓練 ROI: {len(duplicates)} 組(其中分在不同折 {sum(a[2] != a[3] for a in duplicates)} 組)")
    for d in duplicates[:10]:
        print(f"  {d[0]}(fold {d[2]})與 {d[1]}(fold {d[3]}),全景片 {d[4]}")
    print(f"同一張全景片的訓練 ROI 被分到不同折: {len(split)} 張全景片")
    with open(OUT_DIR / "roi_duplicates.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stem_a", "stem_b", "fold_a", "fold_b", "pano_group"])
        writer.writerows(duplicates)
    unresolved = [r["stem"] for r in records if not r["confident"]]
    print(f"\n沒有可信來源的 ROI: {len(unresolved)} 張(前 10: {unresolved[:10]})")
    print(f"逐張來源: {OUT_DIR / 'roi_sources.csv'}\n重複清單: {OUT_DIR / 'roi_duplicates.csv'}")


if __name__ == "__main__":
    main()
