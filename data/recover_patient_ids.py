import argparse
import csv
import json
import re
from collections import Counter

import numpy as np

from stage2.compare_yolo_unet import NAMES, ROOT

# 訓練資料的檔名已被改成 first_NNN / second_NNN,病人編號遺失。
# 同一張影像不論做過哪種前處理,標註的多邊形都相同,所以用標註內容去比對其他資料夾中仍保有原始檔名的標註,
# 還原每張影像的原始編號與左右側。
# 原始編號在兩批資料中各自從頭編,同一個號碼在兩批中是不同的病人,因此病人 = 批次 + 編號:
#   first_NNN  -> 第一批
#   second_NNN -> 第二批(原始檔名帶有 0827_ 前綴,但部分資料夾把前綴拿掉了)
TRAIN_DIR = "datasets/20260909/mf_559__unsharp_mask"
SEARCH_DIRS = ["dental_project", "datasets"]
TEST_LABELS = "runs/gt/labels"
FOLDS_CSV = ROOT / "compare_results" / "kfold_yolo_unet_mf_559" / "folds.csv"
OUT_CSV = ROOT / "patient_map.csv"
TRAIN_STEM = re.compile(r"^(?P<batch>first|second)_\d+$")
# 原始檔名: [first_|second_|0827_]<編號>_<Left|Right>_roi
ORIGINAL_NAME = re.compile(r"^(?P<prefix>(?:(?:first|second|0827)_)*)(?P<number>\d+)_(?P<side>Left|Right)_roi$")
# 不同轉檔程式的小數尾數可能不同,座標差在這個範圍內視為相同
TOLERANCE = 1e-4
CLASS_IDS = {name: i for i, name in NAMES.items()}


def read_txt(path):
    shapes = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 7:
            shapes.append((int(parts[0]), np.array(parts[1:], dtype=np.float64)))
    return shapes


def read_labelme(path):
    try:
        data = json.loads(path.read_text())
        scale = np.array([data["imageWidth"], data["imageHeight"]], dtype=np.float64)
        return [(CLASS_IDS[s["label"]], (np.array(s["points"], dtype=np.float64) / scale).ravel()) for s in data["shapes"]]
    except (KeyError, TypeError, ValueError):
        # 不是 LabelMe 標註或類別不在 NAMES 中
        return []


def structure(shapes):
    # 各多邊形的類別與座標數,用來先縮小比對範圍
    return tuple((cls, len(coords)) for cls, coords in shapes)


def same(a, b):
    return all(np.abs(x - y).max() < TOLERANCE for (_, x), (_, y) in zip(a, b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=TRAIN_DIR)
    parser.add_argument("--search", nargs="+", default=SEARCH_DIRS, help="要搜尋原始檔名標註的資料夾")
    args = parser.parse_args()

    # 仍保有原始檔名的標註: {結構: [(原始檔名, 多邊形)]}
    originals = {}
    n_files = 0
    for folder in args.search:
        for path in sorted((ROOT / folder).rglob("*")):
            if path.suffix not in (".txt", ".json") or not ORIGINAL_NAME.match(path.stem):
                continue
            shapes = read_txt(path) if path.suffix == ".txt" else read_labelme(path)
            if shapes:
                originals.setdefault(structure(shapes), []).append((path.stem, shapes))
                n_files += 1
    print(f"找到 {n_files} 個保有原始檔名的標註檔")

    rows, unmatched, conflicts = [], [], []
    train_labels = sorted(p for p in (ROOT / args.train).rglob("*.txt") if p.parent.name == "labels")
    for path in train_labels:
        stem = TRAIN_STEM.match(path.stem)
        if not stem:
            raise SystemExit(f"{path}: 檔名不是 first_NNN 或 second_NNN,無法判斷批次")
        shapes = read_txt(path)
        names = sorted({name for name, other in originals.get(structure(shapes), []) if same(shapes, other)})
        if not names:
            unmatched.append(path.stem)
            continue
        # 同一張影像在各資料夾的檔名只差在前綴,編號與左右側必須一致
        keys = {(int(ORIGINAL_NAME.match(n)["number"]), ORIGINAL_NAME.match(n)["side"]) for n in names}
        if len(keys) > 1:
            conflicts.append((path.stem, names))
            continue
        number, side = keys.pop()
        rows.append(
            {
                "stem": path.stem,
                "batch": stem["batch"],
                "number": number,
                "side": side,
                "patient": f"{stem['batch']}_{number}",
                "has_0827_name": any("0827_" in n for n in names),
                "originals": " ".join(names),
            }
        )

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n訓練資料 {len(train_labels)} 張: 成功還原 {len(rows)} 張, 找不到對應 {len(unmatched)} 張, 編號互相矛盾 {len(conflicts)} 張")
    if unmatched:
        print(f"  找不到對應(前10): {unmatched[:10]}")
    if conflicts:
        print(f"  編號互相矛盾(前5): {conflicts[:5]}")

    # 檢查「second 批 = 0827 批」這個前提: first 批不應出現 0827_ 的檔名
    for batch in ("first", "second"):
        items = [r for r in rows if r["batch"] == batch]
        print(f"  {batch} 批 {len(items)} 張, 其中 {sum(r['has_0827_name'] for r in items)} 張有對應到 0827_ 開頭的原始檔名")

    patients = Counter(r["patient"] for r in rows)
    print(f"\n共 {len(patients)} 位病人: " + ", ".join(f"{n} 張影像的有 {c} 位" for n, c in sorted(Counter(patients.values()).items())))
    repeated = Counter((r["patient"], r["side"]) for r in rows)
    repeated = [key for key, n in repeated.items() if n > 1]
    if repeated:
        print(f"  [注意] 有 {len(repeated)} 組「同一病人同一側」出現兩次以上(前5): {repeated[:5]}")

    # 目前以影像為單位的切分,有多少病人的影像被分到不同折
    if FOLDS_CSV.exists():
        with open(FOLDS_CSV, newline="") as f:
            fold_of = {r["stem"]: r["fold"] for r in csv.DictReader(f)}
        folds = {}
        for r in rows:
            if r["stem"] in fold_of:
                folds.setdefault(r["patient"], set()).add(fold_of[r["stem"]])
        split = [p for p, fs in folds.items() if len(fs) > 1]
        leaked = sum(patients[p] for p in split)
        multi = sum(n > 1 for n in patients.values())
        print(f"目前的 K 折切分中, {multi} 位有多張影像的病人裡有 {len(split)} 位被分到不同折, 涉及 {leaked} 張影像 ({leaked / len(rows):.1%})")

    # 獨立測試集的檔名沒有批次資訊,分別列出與兩批訓練資料編號相同者
    test = {}
    for path in (ROOT / TEST_LABELS).glob("*.txt"):
        m = ORIGINAL_NAME.match(path.stem)
        if m:
            test.setdefault(int(m["number"]), set()).add(m["side"])
    print(f"\n測試集 {sum(len(s) for s in test.values())} 張, {len(test)} 個編號 (範圍 {min(test)}~{max(test)})")
    for batch in ("first", "second"):
        train = {}
        for r in rows:
            if r["batch"] == batch:
                train.setdefault(r["number"], set()).add(r["side"])
        shared = sorted(set(test) & set(train))
        same_side = [n for n in shared if test[n] & train[n]]
        print(f"  與 {batch} 批編號相同: {len(shared)} 個, 其中同一側在兩邊都出現的有 {len(same_side)} 個")
        if shared:
            print("    " + ", ".join(f"{n}(訓練 {'/'.join(sorted(train[n]))}, 測試 {'/'.join(sorted(test[n]))})" for n in shared[:8]))
    print(f"\n對照表: {OUT_CSV}")


if __name__ == "__main__":
    main()
