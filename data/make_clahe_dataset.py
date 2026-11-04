"""2026-11-03 前處理比較在新切分上重跑: 產生 CLAHE 版的訓練資料集與測試影像(kfold_pano.py --variant clahe 用)。

- 來源與主資料集相同的原圖: 訓練 datasets/20260909/mf_559_original、測試 dental_project/20261008/test。
  主資料集 mf_559__unsharp_mask 與測試 runs/gt/images/um_test 都是由這兩份原圖做 unsharp mask(sigma 1.0、strength 0.5、threshold 3)
  再 cv2.imwrite 存成 JPEG;這裡先重現一次確認來源正確,再換成 CLAHE。
- CLAHE 參數與 kfold_preprocessing.py 相同(clipLimit 2.0、tileGridSize 8×8),同樣以 cv2.imwrite 存 JPEG;標註直接複製。
"""
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from data.mf_unsharp_mask import unsharp_mask

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SRC = ROOT / "datasets/20260909/mf_559_original"
TRAIN_UM = ROOT / "datasets/20260909/mf_559__unsharp_mask"
TEST_SRC = ROOT / "dental_project/20261008/test"
TEST_UM = ROOT / "runs/gt/images/um_test"
TRAIN_OUT = ROOT / "datasets/20261103/mf_559_clahe"
TEST_OUT = ROOT / "runs/gt/images/clahe_test"
UNSHARP = dict(sigma=1.0, strength=0.5, threshold=3)
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def check_source(pairs, name):
    # 原圖 → unsharp mask → JPEG 編解碼,應與現在使用的影像幾乎相同(只差 JPEG 編碼的捨入)
    errors = []
    for src, um in pairs:
        image = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        ok, enc = cv2.imencode(".jpg", unsharp_mask(image, **UNSHARP))
        errors.append(np.abs(cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE).astype(int) - cv2.imread(str(um), cv2.IMREAD_GRAYSCALE).astype(int)).mean())
    print(f"{name}: {len(pairs)} 張,重現 unsharp mask 的平均絕對差 {np.mean(errors):.4f}(最大 {np.max(errors):.4f})")
    if np.max(errors) > 0.05:
        raise SystemExit(f"{name}: 來源原圖與現在的影像對不上,停止")


def main():
    train = sorted(p for p in TRAIN_SRC.rglob("*.jpg") if p.parent.name == "images")
    test = sorted(TEST_SRC.glob("*.jpg"))
    if len(train) != 559 or len(test) != 138:
        raise SystemExit(f"影像數不對: 訓練 {len(train)}、測試 {len(test)}")
    um = {p.name: p for p in TRAIN_UM.rglob("*.jpg") if p.parent.name == "images"}
    check_source([(p, um[p.name]) for p in train], "訓練")
    check_source([(p, TEST_UM / p.name) for p in test], "測試")
    if TRAIN_OUT.exists() or TEST_OUT.exists():
        raise SystemExit(f"{TRAIN_OUT} 或 {TEST_OUT} 已存在,不覆蓋")

    # 資料夾配置(train / valid)照主資料集,kfold_pano.py --variant 以相對路徑對應(mf_559_original 有 90 張放在不同的資料夾)
    for src in train:
        rel = um[src.name].relative_to(TRAIN_UM)
        out = TRAIN_OUT / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), CLAHE.apply(cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)))
        label = TRAIN_UM / rel.parent.parent / "labels" / f"{src.stem}.txt"
        (out.parent.parent / "labels").mkdir(parents=True, exist_ok=True)
        shutil.copy2(label, out.parent.parent / "labels" / label.name)
    TEST_OUT.mkdir(parents=True)
    for src in test:
        cv2.imwrite(str(TEST_OUT / src.name), CLAHE.apply(cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)))
    print(f"完成: {TRAIN_OUT}(559 張 + 標註)、{TEST_OUT}(138 張)")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
