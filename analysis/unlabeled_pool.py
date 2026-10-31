import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from stage2.compare_yolo_unet import ROOT
from data.mf_unsharp_mask import unsharp_mask
from deployment.pipeline_timing import PANO_DIR, UNSHARP

# 沒有標註的 ROI 池(2026-10-28): 20260826__mf_851 的 851 張全景片(= original_pictures 加上 0827_ 前綴)
#   第一階段自動裁出 1521 個 ROI,其中只有一部分有頦孔標註。這裡找出「與任何有標註的影像都無關」的 ROI,
#   用於 (1) 分布檢查(unlabeled_shift.py)與 (2) 半監督訓練(pseudo_label.py)。
#   排除(都是保守做法,多排除只會讓池子變小,不會洩漏):
#     a. 全景片檔名與有標註 ROI 的來源全景片(roi_sources.csv,含訓練與測試)相同
#     b. 全景片縮圖與任一張有標註的來源全景片相關係數 ≥ --pano-thr(檔名不同的重複片,例如 new_pictures/ 與根目錄)
#     c. ROI 縮圖與任一張有標註的 ROI(訓練 559 + 測試 138,含找不到來源的 29 張)相關係數 ≥ --roi-thr
#   影像處理與當初的資料集相同: roi_cropped(存過一次 JPEG)→ unsharp mask(UNSHARP)→ cv2.imwrite 存 JPEG(第二次)。
#   輸出 datasets/20261028/unlabeled_mf851/images/、compare_results/unlabeled/pool.csv(每個自動 ROI 一列與排除原因)。
MF851 = ROOT / "dental_project" / "20260826__mf_851"
ROI_DIR = MF851 / "output" / "roi_cropped"
ROI_SOURCES = ROOT / "compare_results" / "same_patient_check" / "roi_sources.csv"
LABELED_ROI_DIRS = [ROOT / "dental_project" / "20260909" / "mf_559_original" / "images", ROOT / "dental_project" / "20261008" / "test"]
OUT_IMAGES = ROOT / "datasets" / "20261028" / "unlabeled_mf851" / "images"
OUT_CSV = ROOT / "compare_results" / "unlabeled" / "pool.csv"
PREFIX = "0827_"


def thumb(path, size):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    t = cv2.resize(image, size, interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    t -= t.mean()
    return t / (np.linalg.norm(t) + 1e-6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pano-thr", type=float, default=0.95)
    parser.add_argument("--roi-thr", type=float, default=0.95)
    parser.add_argument("--dry-run", action="store_true", help="只算相似度與分布,不輸出影像")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    with open(ROI_SOURCES, newline="") as f:
        sources = {r["source_pano"] for r in csv.DictReader(f) if r["source_pano"]}
    labeled_names = {Path(s).name for s in sources}
    labeled_panos = np.stack([thumb(PANO_DIR / s, (128, 64)) for s in sorted(sources)])
    labeled_rois = np.stack([thumb(p, (48, 48)) for d in LABELED_ROI_DIRS for p in sorted(d.glob("*.jpg"))])
    print(f"有標註: 來源全景片 {len(labeled_panos)} 張、ROI {len(labeled_rois)} 張")

    pano_sim = {}
    for path in sorted(MF851.glob(f"{PREFIX}*.jpg")):
        pano_sim[path.stem] = float((labeled_panos @ thumb(path, (128, 64))).max())

    rows = []
    for path in sorted(ROI_DIR.glob("*.jpg")):
        pano = PREFIX + path.stem.rsplit("_", 2)[0]  # 1000_Left_roi → 全景片 0827_1000
        roi_sim = float((labeled_rois @ thumb(path, (48, 48))).max())
        if pano[len(PREFIX):] + ".jpg" in labeled_names:
            reason = "same_name"
        elif pano_sim[pano] >= args.pano_thr:
            reason = "similar_pano"
        elif roi_sim >= args.roi_thr:
            reason = "similar_roi"
        else:
            reason = ""
        rows.append({"stem": path.stem, "pano": pano, "pano_sim": round(pano_sim[pano], 4), "roi_sim": round(roi_sim, 4), "excluded": reason})

    # 相似度分布: 檔名不同的全景片 / ROI 中,高相似度應該很少(真正的重複片接近 1)
    other = [r for r in rows if r["excluded"] != "same_name"]
    for key in ("pano_sim", "roi_sim"):
        values = np.array([r[key] for r in other])
        print(f"{key}(檔名不同的 {len(values)} 個 ROI): " + ", ".join(f"≥{t}: {int((values >= t).sum())}" for t in (0.8, 0.9, 0.93, 0.95, 0.97, 0.99)))
    pool = [r for r in rows if not r["excluded"]]
    counts = {k: sum(r["excluded"] == k for r in rows) for k in ("same_name", "similar_pano", "similar_roi")}
    print(f"自動 ROI {len(rows)} 個;排除 {counts};池子 {len(pool)} 個 ROI、{len({r['pano'] for r in pool})} 張全景片")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.dry_run:
        return
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    for r in pool:
        roi = cv2.imread(str(ROI_DIR / f"{r['stem']}.jpg"))
        cv2.imwrite(str(OUT_IMAGES / f"{r['stem']}.jpg"), unsharp_mask(roi, **UNSHARP))
    print(f"影像: {OUT_IMAGES}({len(pool)} 張);清單: {OUT_CSV}")


if __name__ == "__main__":
    main()
