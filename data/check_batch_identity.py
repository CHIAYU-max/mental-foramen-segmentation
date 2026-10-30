import csv
import re
from collections import defaultdict

import cv2
import numpy as np

from stage2.compare_yolo_unet import GT_DIR, IMAGES_DIR, ROOT, index_files, load_gt
from data.find_same_patient import find_pano
from stage2.kfold_patient import OUT_DIR as PATIENT_DIR
from stage2.kfold_patient import PATIENT_MAP, UNET_TAG, YOLO_TAG
from stage2.kfold_yolo_unet import K_FOLDS, SEED
from analysis.paper_metrics import collect
from analysis.seed_replicates import OUT_DIR as SEED_DIR
from analysis.seed_replicates import SEEDS

# find_same_patient.py 發現部分測試全景片與第一批訓練全景片完全相同(0827_N 與 N)。
# 這裡逐一比對所有用到的全景片,找出「完全相同的影像」,據此:
#   1. 合併同一位病人(不論批次與編號前綴)
#   2. 檢查訓練資料內是否有同一病人被分到不同折、或同一側的 ROI 重複出現(兩次標註)
#   3. 列出與訓練資料重疊的測試病人,並在排除他們後重算測試集指標
OUT_DIR = ROOT / "compare_results" / "same_patient_check"
SIDE = re.compile(r"^(?P<number>\d+)_(?P<side>Left|Right)_roi$")
# 縮小後影像的相關係數高於此值視為同一張全景片
IDENTICAL = 0.98


def fingerprint(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    image = cv2.equalizeHist(cv2.resize(image, (128, 64), interpolation=cv2.INTER_AREA)).astype(np.float64).ravel()
    return (image - image.mean()) / (image.std() + 1e-6)


def main():
    with open(PATIENT_MAP, newline="") as f:
        info = {r["stem"]: r for r in csv.DictReader(f)}
    with open(PATIENT_DIR / "folds.csv", newline="") as f:
        fold_of = {r["stem"]: int(r["fold"]) for r in csv.DictReader(f)}
    test_stems = sorted(s for s in index_files((ROOT / IMAGES_DIR).resolve()) if SIDE.match(s))

    # 每位(批次, 編號)的全景片
    keys = sorted({(r["batch"], r["number"]) for r in info.values()} | {("test", SIDE.match(s)["number"]) for s in test_stems})
    panos = {k: find_pano("second" if k[0] == "test" else k[0], k[1]) for k in keys}
    missing = [k for k, p in panos.items() if p is None]
    keys = [k for k in keys if panos[k] is not None]
    prints = np.stack([fingerprint(panos[k]) for k in keys])
    corr = prints @ prints.T / prints.shape[1]
    np.fill_diagonal(corr, -1)
    print(f"比對 {len(keys)} 張全景片" + (f";找不到 {len(missing)} 張(前5: {missing[:5]})" if missing else ""))
    off = np.sort(corr[np.triu_indices(len(keys), 1)])
    print(f"兩兩相關係數: 中位數 {np.median(off):.3f}、第 99.9 百分位 {np.percentile(off, 99.9):.3f};高於 {IDENTICAL} 的配對 {int((off > IDENTICAL).sum())} 組")

    # 以「完全相同的全景片」合併病人(union-find);同一張全景片在測試與第二批會有相同的檔案,先合併
    parent = {k: k for k in keys}

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for i, j in zip(*np.where(np.triu(corr > IDENTICAL, 1))):
        parent[root(keys[i])] = root(keys[j])
    for k in keys:
        if k[0] == "test" and ("second", k[1]) in parent:
            parent[root(k)] = root(("second", k[1]))
    merged = {k: root(k) for k in keys}
    groups = defaultdict(list)
    for k in keys:
        groups[merged[k]].append(k)
    multi = [g for g in groups.values() if len(g) > 1]
    print(f"\n完全相同的全景片組成 {len(multi)} 組同一病人(跨批次、跨編號或跨訓練與測試)")
    pairs_by_type = defaultdict(int)
    for g in multi:
        pairs_by_type[" + ".join(sorted({x[0] for x in g}))] += 1
    for t, n in sorted(pairs_by_type.items()):
        print(f"  {t}: {n} 組")

    # 訓練資料:合併後的病人
    train_patient = {s: merged.get((r["batch"], r["number"]), (r["batch"], r["number"])) for s, r in info.items()}
    by_patient = defaultdict(list)
    for s, p in train_patient.items():
        by_patient[p].append(s)
    split = [p for p, ss in by_patient.items() if len({fold_of[s] for s in ss if s in fold_of}) > 1]
    duplicate_sides = [(p, side) for p, ss in by_patient.items() for side in ("Left", "Right") if sum(info[s]["side"] == side for s in ss) > 1]
    n_old = len({r["patient"] for r in info.values()})
    print(f"\n訓練資料: 原本 {n_old} 位病人,合併後 {len(by_patient)} 位")
    print(f"  合併後的病人被分到不同折: {len(split)} 位,涉及 {sum(len(by_patient[p]) for p in split)} 張")
    print(f"  同一病人同一側出現兩次(同一區域被標註兩次): {len(duplicate_sides)} 組")

    # 測試集:與訓練資料重疊的病人
    train_roots = set(train_patient.values())
    contaminated = []
    for s in test_stems:
        m = SIDE.match(s)
        key = ("test", m["number"])
        if key in merged and merged[key] in train_roots:
            same_side = any(info[t]["side"] == m["side"] for t in by_patient[merged[key]])
            contaminated.append((s, same_side))
    print(f"\n測試集 {len(test_stems)} 張中,有 {len(contaminated)} 張的病人也在訓練資料中(其中同一側 {sum(x[1] for x in contaminated)} 張)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "merged_patients.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stem", "batch", "number", "side", "merged_patient", "fold"])
        for s, r in sorted(info.items()):
            writer.writerow([s, r["batch"], r["number"], r["side"], "_".join(train_patient[s]), fold_of.get(s, "")])
    with open(OUT_DIR / "test_overlap.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test_stem", "same_side_in_training"])
        writer.writerows(contaminated)

    # 排除重疊的測試病人後重算測試集(五折多數決)
    bad = {s for s, _ in contaminated}
    gt_dir = (ROOT / GT_DIR).resolve()
    images = index_files((ROOT / IMAGES_DIR).resolve())
    samples = []
    for s in test_stems:
        shape = cv2.imread(str(images[s]), cv2.IMREAD_GRAYSCALE).shape
        samples.append((s, load_gt(s, shape, gt_dir, {})))
    clean = np.array([s not in bad for s, _ in samples])
    print(f"\n測試集(五折多數決),排除重疊病人前後")
    print(f"{'模型':<8}{'種子':<8}{'組別':<12}{'n':>5}{'mf Dice':>10}{'命中 1R':>10}{'ian Dice':>10}")
    for name, tag in (("U-Net", UNET_TAG), ("YOLO", YOLO_TAG)):
        for seed in [SEED] + SEEDS:
            root_dir, variant = (PATIENT_DIR, tag) if seed == SEED else (SEED_DIR, f"{tag}_seed{seed}")
            rows = collect(root_dir, variant, "test", [], samples, True)
            if rows is None:
                continue
            for label, mask in (("全部", np.ones(len(rows), bool)), ("排除重疊後", clean), ("只有重疊", ~clean)):
                sub = [r for r, m in zip(rows, mask) if m]
                if not sub:
                    continue
                hits = [r["mf_hit_1R"] for r in sub if "mf_hit_1R" in r]
                print(f"{name:<8}{seed:<8}{label:<12}{len(sub):>5}{np.mean([r['mf_dice'] for r in sub]):>10.4f}{np.mean(hits):>10.1%}{np.mean([r['ian_dice'] for r in sub]):>10.4f}")
    print(f"\n逐張對照: {OUT_DIR / 'merged_patients.csv'}、{OUT_DIR / 'test_overlap.csv'}")


if __name__ == "__main__":
    main()
