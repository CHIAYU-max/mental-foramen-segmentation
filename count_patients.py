import csv
import re
from collections import Counter, defaultdict

from compare_yolo_unet import IMAGES_DIR, ROOT, index_files
from kfold_pano import EXCLUDE_TEST, ROI_SOURCES
from kfold_patient import PATIENT_MAP

# 核對論文的病人數(2026-10-26,牙科待辦第 3 項):
#   舊稿寫「559 張 ROI、341 位病人」,但 roi_sources.csv 只對到 294 張來源全景片。
#   這裡列出 patient_map.csv 的病人編號、來源全景片、kfold_pano 的分組三者的對應,找出差距從哪裡來。
#   只讀 CSV,不改任何資料。
SIDE = re.compile(r"_(Left|Right)_roi$")
FOLDS_CSV = ROOT / "compare_results" / "kfold_pano_mf_559" / "folds.csv"


def main():
    with open(FOLDS_CSV, newline="") as f:
        folds = {r["stem"]: r for r in csv.DictReader(f)}
    with open(PATIENT_MAP, newline="") as f:
        pmap = {r["stem"]: r for r in csv.DictReader(f)}
    with open(ROI_SOURCES, newline="") as f:
        sources = {r["stem"]: r for r in csv.DictReader(f)}
    stems = sorted(folds)

    patients = {s: pmap[s]["patient"] for s in stems}
    panos = {s: sources[s]["pano_group"] for s in stems if sources.get(s, {}).get("pano_group")}
    print(f"訓練 / CV: {len(stems)} 張 ROI")
    print(f"  patient_map 的病人編號: {len(set(patients.values()))} 個")
    for batch in ("first", "second"):
        sub = [s for s in stems if pmap[s]["batch"] == batch]
        print(f"    {batch:<7} {len(sub)} 張 ROI,{len(set(patients[s] for s in sub))} 個病人編號,"
              f"對到來源全景片 {sum(s in panos for s in sub)} 張 ROI / {len(set(panos[s] for s in sub if s in panos))} 張全景片")
    print(f"  對到來源全景片: {len(panos)} 張 ROI → {len(set(panos.values()))} 張全景片;沒對到 {len(stems) - len(panos)} 張 ROI")
    print(f"  kfold_pano 的組(病人 ∪ 全景片): {len(set(r['group'] for r in folds.values()))} 組")

    # 一張全景片對到幾個病人編號(應該是 1)
    ids_per_pano = defaultdict(set)
    for s, p in panos.items():
        ids_per_pano[p].add(patients[s])
    multi = {p: ids for p, ids in ids_per_pano.items() if len(ids) > 1}
    print(f"\n  同一張全景片卻有不同病人編號: {len(multi)} 張全景片"
          f"(多出的編號共 {sum(len(v) - 1 for v in multi.values())} 個)")
    for p, ids in list(multi.items())[:5]:
        print(f"    {p}: {sorted(ids)}")

    # 一個病人編號對到幾張全景片(> 1 = 同一人不同次拍攝,或編號給錯)
    panos_per_id = defaultdict(set)
    for s, p in panos.items():
        panos_per_id[patients[s]].add(p)
    print(f"  同一個病人編號對到多張全景片: {sum(len(v) > 1 for v in panos_per_id.values())} 個編號")

    # 每張全景片裁出幾個 ROI(左右各一 = 2)
    rois = Counter(panos.values())
    print(f"  每張全景片的 ROI 數分布: {dict(sorted(Counter(rois.values()).items()))}")

    # 病人編號的 ROI 數分布
    per_id = Counter(patients.values())
    print(f"  每個病人編號的 ROI 數分布: {dict(sorted(Counter(per_id.values()).items()))}")

    # 測試集
    test = [s for s in index_files((ROOT / IMAGES_DIR).resolve()) if s not in EXCLUDE_TEST]
    bases = {SIDE.sub("", s) for s in test}
    test_ids = {pmap[s]["patient"] for s in test if s in pmap}
    print(f"\n測試集: {len(test)} 張 ROI,去掉左右側後 {len(bases)} 張來源影像"
          + (f",patient_map 有 {len(test_ids)} 個病人編號" if test_ids else ",不在 patient_map 中"))


if __name__ == "__main__":
    main()
