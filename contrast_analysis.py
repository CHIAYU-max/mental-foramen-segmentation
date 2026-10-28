import argparse
import csv
import statistics

import numpy as np

from kfold_patient import OUT_DIR as PATIENT_DIR
from kfold_patient import UNET_TAG, YOLO_TAG

# 第二批的頦孔明顯比較不暗(check_labels.py 的「頦孔暗度」)。
# 把影像依頦孔暗度分組,在暗度相近的影像之間比較兩批的分數:
#   - 暗度相同時兩批分數接近: 第二批差主要是因為影像本身較不清楚
#   - 暗度相同時第二批仍然較差: 除了影像之外,還有標註方式的差異
LABEL_CHECK = PATIENT_DIR.parent / "label_check" / "label_check.csv"
N_BINS = 4


def main():
    parser = argparse.ArgumentParser()
    # mf_darkness 以 GT 輪廓計算,會受標註畫得寬或窄影響;mf_core_darkness 只用 GT 中心位置,與輪廓無關
    parser.add_argument("--measure", default="mf_core_darkness", choices=["mf_darkness", "mf_core_darkness"])
    args = parser.parse_args()

    with open(LABEL_CHECK, newline="") as f:
        info = {r["filename"]: r for r in csv.DictReader(f) if r["batch"] in ("first", "second") and r.get(args.measure)}
    if not info:
        raise SystemExit(f"{LABEL_CHECK} 沒有 {args.measure} 欄位,請先重新執行 check_labels.py")
    with open(PATIENT_DIR / "compare_cv.csv", newline="") as f:
        dice = {(r["variant"], r["filename"]): float(r["dice"]) for r in csv.DictReader(f) if r["class"] == "mf"}

    stems = sorted(info)
    darkness = np.array([float(info[s][args.measure]) for s in stems])
    print(f"暗度指標: {args.measure}")
    area = np.array([float(info[s]["mf_area"]) for s in stems])
    edges = np.percentile(darkness, np.linspace(0, 100, N_BINS + 1))
    bins = np.clip(np.searchsorted(edges, darkness, side="right") - 1, 0, N_BINS - 1)

    for name, variant in (("U-Net", UNET_TAG), ("YOLO", YOLO_TAG)):
        scores = np.array([dice[(variant, s)] for s in stems])
        print(f"\n===== {name}: 頦孔 Dice 依頦孔暗度分組(暗度越高代表頦孔越明顯) =====")
        print(f"{'暗度範圍':<16}" + "".join(f"{b + ' n':>10}{b + ' Dice':>14}" for b in ("first", "second")) + f"{'差距':>10}")
        for i in range(N_BINS):
            text = f"{edges[i]:>6.2f} ~ {edges[i + 1]:<6.2f}  "
            means = {}
            for batch in ("first", "second"):
                idx = [j for j, s in enumerate(stems) if bins[j] == i and info[s]["batch"] == batch]
                means[batch] = statistics.mean(scores[idx]) if idx else float("nan")
                text += f"{len(idx):>10}{means[batch]:>14.4f}"
            print(text + f"{means['first'] - means['second']:>+10.4f}")
        # 整體: 暗度與 Dice 的相關,以及加入批次後批次本身還剩多少影響(線性迴歸)
        is_second = np.array([info[s]["batch"] == "second" for s in stems], dtype=float)
        X = np.column_stack([np.ones(len(stems)), darkness, np.log(area), is_second])
        coef, *_ = np.linalg.lstsq(X, scores, rcond=None)
        raw_gap = scores[is_second == 0].mean() - scores[is_second == 1].mean()
        print(f"暗度與 Dice 的相關係數 {np.corrcoef(darkness, scores)[0, 1]:.3f}")
        print(f"兩批的原始差距 {raw_gap:+.4f};控制暗度與頦孔面積後,批次本身的差距 {-coef[3]:+.4f}")


if __name__ == "__main__":
    main()
