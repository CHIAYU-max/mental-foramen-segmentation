import argparse
import csv
import json
import random
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

from compare_preprocessing import collect_images
from compare_yolo_unet import GT_DIR, IMAGES_DIR, ROOT, index_files, load_gt
from kfold_patient import DATASET, PATIENT_MAP, UNET_TAG, YOLO_TAG
from kfold_patient import OUT_DIR as PATIENT_DIR
from paper_metrics import bootstrap, collect

# 判定每張影像的頦孔是否可辨識(二分法),再依此分層報告模型表現。
#   prepare: 把訓練 559 張與測試 138 張以匿名、隨機順序複製到 OUT_DIR/images,對照表另存
#   analyze: 讀取判定結果,產出分層的結果,並與 annotation_agreement.py 的重新標註比較判定的一致性
# 判定方式(擇一):
#   1. LabelMe: labelme visibility_grading/images --flags visible,not_visible --autosave --nodata
#      每張勾選 visible 或 not_visible 其中一個
#   2. 直接在 OUT_DIR/grades.csv 的 mf_visible 欄填 1(看得出來)或 0(看不出來)
OUT_DIR = ROOT / "visibility_grading"
IMAGES = OUT_DIR / "images"
KEY_CSV = OUT_DIR / "key.csv"
GRADES_CSV = OUT_DIR / "grades.csv"
SEED = 0
SIDE = re.compile(r"_(Left|Right)_roi$")
MODELS = [("U-Net", UNET_TAG), ("YOLO", YOLO_TAG)]


def prepare(args):
    if KEY_CSV.exists() and not args.force:
        raise SystemExit(f"{KEY_CSV} 已存在;為避免蓋掉進行中的判定而中止,確定要重新產生請加 --force")
    with open(PATIENT_MAP, newline="") as f:
        batch_of = {r["stem"]: r["batch"] for r in csv.DictReader(f)}
    items = [(s, p, "cv", batch_of[s]) for s, p in collect_images((ROOT / DATASET).resolve()).items()]
    items += [(s, p, "test", "test") for s, p in index_files((ROOT / IMAGES_DIR).resolve()).items()]
    random.Random(SEED).shuffle(items)

    IMAGES.mkdir(parents=True, exist_ok=True)
    with open(KEY_CSV, "w", newline="") as f, open(GRADES_CSV, "w", newline="") as g:
        key, grades = csv.writer(f), csv.writer(g)
        key.writerow(["alias", "stem", "set", "batch", "source"])
        grades.writerow(["alias", "mf_visible"])
        for i, (stem, path, dataset, batch) in enumerate(items, start=1):
            alias = f"v{i:04d}"
            shutil.copy2(path, IMAGES / f"{alias}{path.suffix}")
            key.writerow([alias, stem, dataset, batch, path])
            grades.writerow([alias, ""])
    print(f"已複製 {len(items)} 張影像到 {IMAGES}(匿名、隨機順序)")
    print("判定時請不要打開 key.csv,也不要參考舊標註或模型預測")


def read_grades():
    # 優先讀 grades.csv 已填的值,其次讀 LabelMe 的 flags
    grades = {}
    with open(GRADES_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if r["mf_visible"].strip() in ("0", "1"):
                grades[r["alias"]] = int(r["mf_visible"])
    conflicts = []
    for path in IMAGES.glob("*.json"):
        flags = json.loads(path.read_text()).get("flags", {})
        visible, hidden = flags.get("visible", False), flags.get("not_visible", False)
        if visible == hidden:
            if visible:
                conflicts.append(path.stem)
            continue
        grades.setdefault(path.stem, int(visible))
    if conflicts:
        print(f"[注意] {len(conflicts)} 張同時勾選了 visible 與 not_visible,未列入(前5: {conflicts[:5]})")
    return grades


def kappa(a, b):
    a, b = np.array(a), np.array(b)
    observed = (a == b).mean()
    expected = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return observed, (observed - expected) / (1 - expected) if expected < 1 else float("nan")


def analyze(args):
    with open(KEY_CSV, newline="") as f:
        key = {r["alias"]: r for r in csv.DictReader(f)}
    grades = read_grades()
    missing = sorted(set(key) - set(grades))
    print(f"已判定 {len(grades)} / {len(key)} 張" + (f",尚未判定(前5): {missing[:5]}" if missing else ""))
    visible = {key[a]["stem"]: v for a, v in grades.items()}

    print("\n===== 頦孔可辨識的比例 =====")
    for group in ("first", "second", "test"):
        values = [v for a, v in grades.items() if key[a]["batch"] == group]
        if values:
            print(f"  {group:<8} {sum(values)} / {len(values)} ({np.mean(values):.1%})")

    # 與重新標註時的判斷比較(同一批影像在兩次判定中是否一致)
    agreement_csv = ROOT / "annotation_agreement" / "agreement.csv"
    if agreement_csv.exists():
        with open(agreement_csv, newline="") as f:
            earlier = {r["filename"]: int(int(r["new_area"]) > 0) for r in csv.DictReader(f) if r["class"] == "mf"}
        common = sorted(set(earlier) & set(visible))
        if common:
            observed, k = kappa([earlier[s] for s in common], [visible[s] for s in common])
            print(f"\n與重新標註時的判斷比較: {len(common)} 張,一致率 {observed:.1%},Cohen's kappa {k:.2f}")

    with open(PATIENT_MAP, newline="") as f:
        patient_of = {r["stem"]: r["patient"] for r in csv.DictReader(f)}
    with open(PATIENT_DIR / "folds.csv", newline="") as f:
        fold_of = {r["stem"]: int(r["fold"]) for r in csv.DictReader(f)}
    by_stem = {r["stem"]: r for r in key.values()}
    rng = np.random.default_rng(0)

    def report(title, samples, rows_of, groups, patients):
        print(f"\n===== {title} =====")
        print(f"{'組別':<22}{'n':>5}" + "".join(f"{name + ' mf Dice (95% CI)':>32}{name + ' 命中1R':>14}" for name, _ in MODELS))
        for label in ("可辨識", "無法辨識", "全部"):
            idx = [i for i, g in enumerate(groups) if label == "全部" or g == label]
            if not idx:
                continue
            text = f"{label:<22}{len(idx):>5}"
            for name, _ in MODELS:
                rows = [rows_of[name][i] for i in idx]
                est, low, high = bootstrap(rows, [patients[i] for i in idx], "mf_dice", np.mean, rng)
                hits = [r["mf_hit_1R"] for r in rows if "mf_hit_1R" in r]
                text += f"{est:>14.3f} ({low:.3f}-{high:.3f})   {np.mean(hits):>12.1%}"
            print(text)

    # 交叉驗證: 只納入已判定的影像,各批分開與合併
    for batch in ("first", "second", "all"):
        stems = sorted(s for s in visible if s in fold_of and (batch == "all" or by_stem[s]["batch"] == batch))
        samples = []
        for stem in stems:
            path = Path(by_stem[stem]["source"])
            shape = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).shape
            samples.append((fold_of[stem], stem, load_gt(stem, shape, path.parent.parent / "labels", {})))
        if not samples:
            continue
        rows_of = {name: collect(PATIENT_DIR, variant, "cv", samples, [], False) for name, variant in MODELS}
        groups = ["可辨識" if visible[s] else "無法辨識" for s in stems]
        report(f"交叉驗證 / {batch}", samples, rows_of, groups, [patient_of[s] for s in stems])

    # 測試集: 五折多數決
    test_gt = (ROOT / GT_DIR).resolve()
    images = index_files((ROOT / IMAGES_DIR).resolve())
    stems = sorted(s for s in images if s in visible)
    if stems:
        samples = []
        for stem in stems:
            shape = cv2.imread(str(images[stem]), cv2.IMREAD_GRAYSCALE).shape
            samples.append((stem, load_gt(stem, shape, test_gt, {})))
        rows_of = {name: collect(PATIENT_DIR, variant, "test", [], samples, True) for name, variant in MODELS}
        groups = ["可辨識" if visible[s] else "無法辨識" for s in stems]
        report("測試集(五折多數決)", samples, rows_of, groups, [SIDE.sub("", s) for s in stems])

    out_csv = OUT_DIR / "visibility.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stem", "set", "batch", "mf_visible"])
        for alias, v in sorted(grades.items()):
            writer.writerow([key[alias]["stem"], key[alias]["set"], key[alias]["batch"], v])
    print(f"\n各影像的判定結果(已對回原始檔名): {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="匿名複製影像並產生判定表")
    p.add_argument("--force", action="store_true", help="重新產生並覆寫對照表與判定表")
    commands.add_parser("analyze", help="讀取判定結果並產出分層結果")
    args = parser.parse_args()
    {"prepare": prepare, "analyze": analyze}[args.command](args)


if __name__ == "__main__":
    main()
