import csv
import re
import statistics
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from stage2.compare_yolo_unet import GT_DIR, IMAGES_DIR, ROOT, index_files, load_gt
from stage2.kfold_patient import OUT_DIR as PATIENT_DIR
from stage2.kfold_patient import PATIENT_MAP, UNET_TAG, YOLO_TAG
from stage2.kfold_unet_variants import fold_paths
from stage2.kfold_yolo_unet import K_FOLDS
from analysis.paper_metrics import collect

# 找出第一批與第二批表現差距的來源:把交叉驗證(與測試集)的逐張結果依各種條件分組比較。
# 條件: 資料批次、第二批中標註的來源、原始全景片的解析度、左右側、頦孔大小
PANO_DIRS = {"first": ROOT / "dental_project" / "original_pictures", "second": ROOT / "dental_project" / "20260826__mf_851"}
SIDE = re.compile(r"^(?P<number>.+)_(?P<side>Left|Right)_roi$")
MODELS = [("U-Net", UNET_TAG), ("YOLO", YOLO_TAG)]


def pano_size(batch, number):
    # 原始全景片: 第一批為 <編號>(可能沒有副檔名),第二批為 0827_<編號>
    folder = PANO_DIRS[batch]
    stem = str(number) if batch == "first" else f"0827_{number}"
    for path in [folder / stem] + sorted(folder.glob(f"{stem}.*")):
        if path.is_file():
            try:
                return Image.open(path).size
            except OSError:
                pass
    return None


def resolution_group(size):
    if size is None:
        return "找不到原圖"
    return {1032: "高 1032", 1280: "高 1280"}.get(size[1], "其他尺寸")


def table(title, groups, rows_by_model):
    print(f"\n===== {title} =====")
    header = f"{'組別':<28}{'n':>5}"
    for name, _ in MODELS:
        header += f"{name + ' mf Dice':>16}{name + ' 命中1R':>12}"
    print(header + f"{'U-Net ian Dice':>16}")
    for label in sorted(set(groups)):
        idx = [i for i, g in enumerate(groups) if g == label]
        text = f"{label:<28}{len(idx):>5}"
        for name, _ in MODELS:
            rows = [rows_by_model[name][i] for i in idx]
            hits = [r["mf_hit_1R"] for r in rows if "mf_hit_1R" in r]
            text += f"{statistics.mean(r['mf_dice'] for r in rows):>16.4f}{statistics.mean(hits) if hits else float('nan'):>12.1%}"
        text += f"{statistics.mean(rows_by_model['U-Net'][i]['ian_dice'] for i in idx):>16.4f}"
        print(text)


def main():
    with open(PATIENT_MAP, newline="") as f:
        info = {r["stem"]: r for r in csv.DictReader(f)}

    cv_samples, meta = [], []
    for fold in range(K_FOLDS):
        for path in fold_paths(fold, "val", PATIENT_DIR):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            gt = load_gt(path.stem, image.shape, path.parent.parent / "labels", {})
            cv_samples.append((fold, path.stem, gt))
            r = info[path.stem]
            meta.append({"batch": r["batch"], "side": r["side"], "number": r["number"], "has_0827": r["has_0827_name"] == "True", "mf_area": int(gt["mf"].sum())})

    rows_by_model = {name: collect(PATIENT_DIR, variant, "cv", cv_samples, [], False) for name, variant in MODELS}

    table("交叉驗證: 資料批次", [m["batch"] for m in meta], rows_by_model)
    # 第二批的標註來自兩種來源資料夾: 檔名保留 0827_ 前綴的,以及只存在於 mf_262 資料夾、前綴被拿掉的
    table(
        "交叉驗證: 第二批依標註來源",
        [f"second / {'有 0827_ 檔名' if m['has_0827'] else '只在 mf_262'}" if m["batch"] == "second" else "first" for m in meta],
        rows_by_model,
    )
    sizes = [pano_size(m["batch"], m["number"]) for m in meta]
    table("交叉驗證: 批次 × 原始全景片解析度", [f"{m['batch']} / {resolution_group(s)}" for m, s in zip(meta, sizes)], rows_by_model)
    table("交叉驗證: 批次 × 左右側", [f"{m['batch']} / {m['side']}" for m in meta], rows_by_model)
    # 頦孔大小依全體影像的三分位數分組
    cuts = np.percentile([m["mf_area"] for m in meta if m["mf_area"] > 0], [33.3, 66.7])
    table(
        "交叉驗證: 批次 × 頦孔 GT 面積(三分位)",
        [f"{m['batch']} / {'小' if m['mf_area'] <= cuts[0] else '中' if m['mf_area'] <= cuts[1] else '大'}" for m in meta],
        rows_by_model,
    )

    # 測試集全部來自第二批的來源資料夾,依解析度分組看是否與交叉驗證一致
    test_gt = (ROOT / GT_DIR).resolve()
    test_samples, test_groups = [], []
    for stem, path in index_files((ROOT / IMAGES_DIR).resolve()).items():
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        test_samples.append((stem, load_gt(stem, image.shape, test_gt, {})))
        test_groups.append(f"test / {resolution_group(pano_size('second', SIDE.match(stem)['number']))}")
    test_rows = {name: collect(PATIENT_DIR, variant, "test", [], test_samples, True) for name, variant in MODELS}
    table("測試集(五折多數決): 原始全景片解析度", test_groups, test_rows)

    missing = sum(s is None for s in sizes)
    if missing:
        print(f"\n[注意] 交叉驗證有 {missing} 張找不到對應的原始全景片,解析度歸在「找不到原圖」")


if __name__ == "__main__":
    main()
