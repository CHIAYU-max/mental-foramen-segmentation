import argparse
import csv
import re
import sys

import cv2
import numpy as np

from stage2.compare_yolo_unet import GT_DIR, IMAGES_DIR, ROOT, index_files, load_gt
from stage2.kfold_pano import EXCLUDE_TEST, PROJECT
from analysis.paper_metrics import bootstrap
from analysis.pano_stats import per_roi

# 新切分(kfold_pano)上的頦孔可辨識性分層(2026-10-26):
#   舊的 visibility_grading.py 綁定病人層級切分(kfold_patient_mf_559),CV / SOP 引用的
#   「可辨識影像 Dice 0.83,與標註者自身一致性 0.79 相當」是舊切分的數字,這裡在新切分上重算。
#   可辨識性沿用 visibility_grading/visibility.csv 的人工判定(1 = 看得出頦孔)。
#   CV: 每張用沒看過它的那一折;測試: 五折多數決(與 kfold_pano.evaluate 相同)。
#   95% CI: 以「病人 ∪ 來源全景片」的組(CV,folds.csv 的 group)或去掉左右側的來源影像(測試)為單位 bootstrap。
#   標註者自身一致性: annotation_agreement/agreement.csv 中同一人兩次標註的 mf Dice。
VISIBILITY_CSV = ROOT / "visibility_grading" / "visibility.csv"
AGREEMENT_CSV = ROOT / "annotation_agreement" / "agreement.csv"
SIDE = re.compile(r"_(Left|Right)_roi$")
TAGS = ["r34_imagenet_aug", "r34_imagenet_aug_s1", "r34_imagenet_aug_s2", "yolo26s-seg", "yolov8n-seg"]
STRATA = [("visible", 1), ("not_visible", 0), ("all", None)]


def load_test():
    test_gt = (ROOT / GT_DIR).resolve()
    test = []
    for stem, path in index_files((ROOT / IMAGES_DIR).resolve()).items():
        if stem in EXCLUDE_TEST:
            continue
        shape = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).shape
        test.append((stem, load_gt(stem, shape, test_gt, {})))
    return test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", default=TAGS, help="masks 下的模型名稱")
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--post", choices=["none", "lcc"], default="none", help="2026-10-29: 頦孔後處理(同 pano_stats.py --post),輸出檔名加 _lcc")
    args = parser.parse_args()
    suffix = "" if args.post == "none" else f"_{args.post}"
    sys.stdout.reconfigure(line_buffering=True)
    out_dir = ROOT / "compare_results" / args.project

    with open(VISIBILITY_CSV, newline="") as f:
        visible = {r["stem"]: int(r["mf_visible"]) for r in csv.DictReader(f)}
    with open(out_dir / "folds.csv", newline="") as f:
        group_of = {r["stem"]: r["group"] for r in csv.DictReader(f)}

    print("===== 頦孔可辨識的比例 =====")
    for setting, stems in (("cv", sorted(group_of)), ("test", None)):
        if stems is None:
            stems = [s for s, _ in load_test()]
        graded = [visible[s] for s in stems if s in visible]
        print(f"  {setting:<5} {sum(graded)} / {len(graded)} 可辨識 ({np.mean(graded):.1%});未判定 {len(stems) - len(graded)} 張")

    with open(AGREEMENT_CSV, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["class"] == "mf"]
    both = [float(r["old_vs_new"]) for r in rows if int(r["old_area"]) > 0 and int(r["new_area"]) > 0]
    print(f"\n標註者自身一致性(同一人兩次標註的 mf Dice): 兩次都有標頦孔 {len(both)} 張,平均 {np.mean(both):.3f}"
          f";全部 {len(rows)} 張(一次有一次沒有算 0)平均 {np.mean([float(r['old_vs_new']) for r in rows]):.3f}")

    test = load_test()
    rng = np.random.default_rng(0)
    results = []
    for tag in args.tags:
        data = per_roi(tag, out_dir, test, args.post)
        for setting, rows_by_stem in data.items():
            for name, want in STRATA:
                stems = [s for s in sorted(rows_by_stem) if s in visible and (want is None or visible[s] == want)]
                if not stems:
                    continue
                rows = [rows_by_stem[s] for s in stems]
                groups = [group_of[s] if setting == "cv" else SIDE.sub("", s) for s in stems]
                dice = bootstrap(rows, groups, "mf_dice", np.mean, rng)
                hit = bootstrap(rows, groups, "mf_hit_1R", np.mean, rng)
                results.append({"model": tag, "setting": setting, "stratum": name, "n": len(stems),
                                "mf_dice": dice[0], "mf_dice_low": dice[1], "mf_dice_high": dice[2],
                                "mf_hit_1R": hit[0], "mf_hit_low": hit[1], "mf_hit_high": hit[2]})

    print(f"\n{'model':<22}{'setting':<7}{'stratum':<13}{'n':>5}{'mf Dice (95% CI)':>26}{'命中 1R (95% CI)':>26}")
    for r in results:
        print(f"{r['model']:<22}{r['setting']:<7}{r['stratum']:<13}{r['n']:>5}"
              f"{r['mf_dice']:>10.3f} ({r['mf_dice_low']:.3f}-{r['mf_dice_high']:.3f})"
              f"{r['mf_hit_1R']:>10.1%} ({r['mf_hit_low']:.1%}-{r['mf_hit_high']:.1%})")

    # U-Net 三個種子的平均(各層各設定)
    seeds = [t for t in args.tags if t.startswith("r34_imagenet_aug")]
    if len(seeds) > 1:
        print(f"\nU-Net {len(seeds)} 個種子的平均 ± 標準差:")
        for setting in ("cv", "test"):
            for name, _ in STRATA:
                picked = [r for r in results if r["model"] in seeds and r["setting"] == setting and r["stratum"] == name]
                if len(picked) == len(seeds):
                    d = [r["mf_dice"] for r in picked]
                    h = [r["mf_hit_1R"] for r in picked]
                    print(f"  {setting:<5}{name:<13} mf Dice {np.mean(d):.3f} ± {np.std(d, ddof=1):.3f}   命中 {np.mean(h):.1%} ± {np.std(h, ddof=1):.1%}")

    with open(out_dir / f"visibility_strata{suffix}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in results)
    print(f"\n輸出: {out_dir / f'visibility_strata{suffix}.csv'}")


if __name__ == "__main__":
    main()
