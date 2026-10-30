import csv
import sys

import cv2
import numpy as np

from stage2.compare_yolo_unet import ROOT
from data.find_same_patient import OUT_DIR, REVIEW_PAIRS, embed, side_by_side
from stage2.kfold_pano import PROJECT

# 2026-10-30 訓練資料內部的同一病人檢查: find_same_patient.py 只比「測試 × 訓練」,交叉驗證各折之間還沒查過。
#   559 張 ROI 的來源全景片(同 find_same_patient.py,只用比對可靠的來源)兩兩算影像特徵相似度,只列出不同折的配對,
#   前 REVIEW_PAIRS 組並排存到 same_patient_check/review_cv/ 供人工確認,並在 candidates_cv.csv 填 same_person。
#   最壞情況: 把前 REVIEW_PAIRS 組牽涉到的 ROI 全部排除,看 U-Net 與 YOLO26s 的 CV 命中(種子 0、lcc)變多少。
SPLIT_DIR = ROOT / "compare_results" / PROJECT
REVIEW_DIR = OUT_DIR / "review_cv"


def main():
    sys.stdout.reconfigure(line_buffering=True)
    with open(SPLIT_DIR / "folds.csv", newline="") as f:
        folds = list(csv.DictReader(f))
    with open(OUT_DIR / "roi_sources.csv", newline="") as f:
        sources = {r["stem"]: r for r in csv.DictReader(f) if r["set"] == "train" and r["source_pano"]}

    pano_root = ROOT / "dental_project" / "original_pictures"
    panos = {}
    skipped = 0
    for row in folds:
        source = sources.get(row["stem"])
        if source is None or float(source["score"]) < 0.9 or float(source["score"]) - float(source["second_score"] or 0) < 0.06:
            skipped += 1
            continue
        path = pano_root / source["pano_group"]
        if path.is_file():
            entry = panos.setdefault(source["pano_group"], {"path": path, "folds": set(), "stems": []})
            entry["folds"].add(row["fold"])
            entry["stems"].append(row["stem"])
        else:
            skipped += 1
    print(f"來源全景片 {len(panos)} 張;略過 {skipped} 張 ROI(比對不可靠或找不到原圖)")

    keys = list(panos)
    features = embed([panos[k]["path"] for k in keys])
    similarity = features @ features.T
    pairs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if not panos[keys[i]]["folds"] & panos[keys[j]]["folds"]:
                pairs.append((float(similarity[i, j]), keys[i], keys[j]))
    pairs.sort(reverse=True)
    values = np.array([p[0] for p in pairs])
    print(f"不同折的配對 {len(pairs)} 組;相似度 >= 0.99 {int((values >= 0.99).sum())}、>= 0.97 {int((values >= 0.97).sum())}、>= 0.95 {int((values >= 0.95).sum())}")

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "candidates_cv.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "similarity", "pano_a", "folds_a", "pano_b", "folds_b", "image", "same_person(y/n/unsure)"])
        for rank, (s, a, b) in enumerate(pairs[:REVIEW_PAIRS], start=1):
            name = f"{rank:03d}_{s:.3f}.jpg"
            title = f"#{rank}  similarity {s:.3f}  left: {panos[a]['path'].name} (fold {sorted(panos[a]['folds'])})   right: {panos[b]['path'].name} (fold {sorted(panos[b]['folds'])})"
            cv2.imwrite(str(REVIEW_DIR / name), side_by_side(panos[a]["path"], panos[b]["path"], title))
            writer.writerow([rank, round(s, 4), a, sorted(panos[a]["folds"]), b, sorted(panos[b]["folds"]), name, ""])

    affected = {stem for _, a, b in pairs[:REVIEW_PAIRS] for key in (a, b) for stem in panos[key]["stems"]}
    with open(SPLIT_DIR / "per_roi_lcc.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["setting"] == "cv"]
    print(f"前 {REVIEW_PAIRS} 組相似度 {pairs[0][0]:.4f}–{pairs[min(REVIEW_PAIRS, len(pairs)) - 1][0]:.4f},牽涉 {len(affected)} 張 ROI")
    hit = {}
    for model in ("r34_imagenet_aug", "yolo26s-seg"):
        part = [r for r in rows if r["model"] == model]
        rest = [r for r in part if r["stem"] not in affected]
        hit[model] = (np.mean([float(r["mf_hit_1R"]) for r in part]), np.mean([float(r["mf_hit_1R"]) for r in rest]))
        print(f"{model}: 全部 {hit[model][0]:.4f}({len(part)} 張)→ 排除後 {hit[model][1]:.4f}({len(rest)} 張)")
    print(f"差距 {hit['r34_imagenet_aug'][0] - hit['yolo26s-seg'][0]:+.4f} → {hit['r34_imagenet_aug'][1] - hit['yolo26s-seg'][1]:+.4f}")


if __name__ == "__main__":
    main()
