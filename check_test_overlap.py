import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent

# 訓練資料的檔名已被改成 first_NNN,無法用檔名比對,改用標註內容(多邊形座標)判斷是否為同一張影像
TRAIN_DIR = "datasets/20260909/mf_559__unsharp_mask"
SOURCE_DIR = "dental_project/20260909/mf_559__unsharp_mask"
TEST_LABELS = "runs/gt/labels"
# 座標取到小數第 4 位再比對,避免不同轉檔程式的尾數差異
DECIMALS = 4
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def label_key(path):
    lines = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if parts:
            lines.append((parts[0],) + tuple(round(float(v), DECIMALS) for v in parts[1:]))
    return tuple(sorted(lines))


def collect_labels(folder):
    # 回傳 {標註內容: [檔案路徑]},空白標註不列入
    labels = {}
    for p in sorted(folder.rglob("*.txt")):
        if p.parent.name == "labels" and label_key(p):
            labels.setdefault(label_key(p), []).append(p)
    return labels


def find_image(folder, stem):
    for p in folder.rglob(f"{stem}.*"):
        if p.suffix.lower() in IMAGE_SUFFIXES and p.parent.name == "images":
            return p
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=TRAIN_DIR)
    parser.add_argument("--source", default=SOURCE_DIR)
    parser.add_argument("--test-labels", default=TEST_LABELS)
    args = parser.parse_args()

    train_dir = (ROOT / args.train).resolve()
    source_dir = (ROOT / args.source).resolve()
    train = collect_labels(train_dir)
    test = collect_labels((ROOT / args.test_labels).resolve())
    n_train = sum(len(v) for v in train.values())
    n_test = sum(len(v) for v in test.values())

    # 1. 測試集是否出現在訓練資料中
    overlap = [(paths[0], train[key][0]) for key, paths in test.items() if key in train]
    print(f"訓練資料 {n_train} 個標註, 測試集 {n_test} 個標註")
    print(f"測試集中與訓練資料標註相同的影像: {len(overlap)} / {n_test}")
    for test_path, train_path in overlap[:10]:
        print(f"  {test_path.name}  ==  {train_path.relative_to(train_dir)}")

    # 2. 訓練資料內部是否有重複影像(會讓 K-fold 的訓練與驗證互相洩漏)
    duplicates = [paths for paths in train.values() if len(paths) > 1]
    print(f"\n訓練資料內標註完全相同的組數: {len(duplicates)}")
    for paths in duplicates[:10]:
        print("  " + "  ==  ".join(str(p.relative_to(train_dir)) for p in paths))

    # 3. 切分後的影像與來源資料夾的影像是否相同(兩個資料夾的前處理名稱不一致)
    print(f"\n切分後影像與來源影像的像素比較({train_dir.name} vs {source_dir.name})")
    for paths in list(train.values())[:5]:
        stem = paths[0].stem
        a, b = find_image(train_dir, stem), find_image(source_dir, stem)
        if a is None or b is None:
            print(f"  {stem}: 找不到影像(切分後 {a}, 來源 {b})")
            continue
        img_a, img_b = cv2.imread(str(a)), cv2.imread(str(b))
        if img_a.shape != img_b.shape:
            print(f"  {stem}: 尺寸不同 {img_a.shape} vs {img_b.shape}")
        elif np.array_equal(img_a, img_b):
            print(f"  {stem}: 相同")
        else:
            print(f"  {stem}: 不同, 最大差 {np.abs(img_a.astype(int) - img_b).max()}")


if __name__ == "__main__":
    main()
