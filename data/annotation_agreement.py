import argparse
import csv
import random
import shutil
import statistics
from pathlib import Path

import cv2

from stage2.compare_preprocessing import collect_images
from stage2.compare_yolo_unet import NAMES, ROOT, load_gt, pixel_metrics, wilcoxon_p
from stage2.kfold_patient import OUT_DIR as PATIENT_DIR
from stage2.kfold_patient import PATIENT_MAP, UNET_TAG
from analysis.paper_metrics import read_masks

# 重新標註一批影像,用來回答兩個問題:
#   1. 現在的標註方式與當初差多少(新舊標註的一致性),第一批與第二批是否不同
#   2. 兩批的分數差距是否來自舊標註: 比較最終模型的預測與「舊標註」和「新標註」的 Dice
#      若第二批的模型預測與新標註明顯較吻合,代表第二批的舊標註是分數偏低的主因
#   sample: 從兩批各抽一半,把影像(不含標註)以打亂順序的匿名檔名複製到 OUT_DIR/images,標註時看不出來自哪一批
#   score:  重新標註完成後(LabelMe 的 .json 存在影像旁邊),計算上述比較
DATASET = "datasets/20260909/mf_559__unsharp_mask"
OUT_DIR = ROOT / "annotation_agreement"
IMAGES_DIR = OUT_DIR / "images"
SAMPLE_CSV = OUT_DIR / "sample.csv"
N_SAMPLES = 40
SAMPLE_SEED = 0


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def sample(args):
    if SAMPLE_CSV.exists() and not args.force:
        raise SystemExit(f"{SAMPLE_CSV} 已存在;為避免蓋掉進行中的標註而中止,確定要重新抽樣請加 --force")
    images = collect_images((ROOT / args.dataset).resolve())
    batch_of = {r["stem"]: r["batch"] for r in read_csv(PATIENT_MAP)}
    rng = random.Random(args.seed)
    if args.batch == "both":
        # 兩批各抽一半,才能分別比較
        stems = []
        for batch in ("first", "second"):
            group = sorted(s for s in images if batch_of[s] == batch)
            rng.shuffle(group)
            stems += group[: args.n // 2]
    else:
        stems = sorted(s for s in images if args.batch == "all" or batch_of[s] == args.batch)
        rng.shuffle(stems)
        stems = stems[: args.n]
    rng.shuffle(stems)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    with open(SAMPLE_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alias", "stem", "batch", "source"])
        for i, stem in enumerate(stems, start=1):
            alias = f"s{i:03d}"
            shutil.copy2(images[stem], IMAGES_DIR / f"{alias}{images[stem].suffix}")
            writer.writerow([alias, stem, batch_of[stem], images[stem]])
    print(f"已抽出 {len(stems)} 張(seed={args.seed}): " + ", ".join(f"{b} {sum(batch_of[s] == b for s in stems)} 張" for b in ("first", "second")))
    print(f"影像複製到: {IMAGES_DIR}")
    print("請依最終的標註規則,用 LabelMe 重新標註 mf 與 ian(.json 存在影像旁邊),完成後執行 python -m data.annotation_agreement score")


def score(args):
    if not SAMPLE_CSV.exists():
        raise SystemExit(f"找不到 {SAMPLE_CSV},請先執行 python -m data.annotation_agreement sample")
    sample_rows = read_csv(SAMPLE_CSV)
    fold_of = {r["stem"]: int(r["fold"]) for r in read_csv(PATIENT_DIR / "folds.csv")}

    rows, pending = [], []
    for r in sample_rows:
        alias, stem, source = r["alias"], r["stem"], Path(r["source"])
        if not (IMAGES_DIR / f"{alias}.json").exists():
            pending.append(alias)
            continue
        image = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        old = load_gt(stem, image.shape, source.parent.parent / "labels", {})
        new = load_gt(alias, image.shape, IMAGES_DIR, {})
        # 每張影像都用沒看過它的那一折模型的預測
        pred = read_masks(PATIENT_DIR, "cv", UNET_TAG, fold_of[stem], stem)
        for cls in NAMES.values():
            rows.append(
                {
                    "filename": stem,
                    "batch": r["batch"],
                    "class": cls,
                    "old_vs_new": pixel_metrics(new[cls], old[cls])["dice"],
                    "model_vs_old": pixel_metrics(pred[cls], old[cls])["dice"],
                    "model_vs_new": pixel_metrics(pred[cls], new[cls])["dice"],
                    "old_area": int(old[cls].sum()),
                    "new_area": int(new[cls].sum()),
                }
            )
    if pending:
        print(f"尚未重新標註: {len(pending)} / {len(sample_rows)} 張(前5: {pending[:5]})")
    if not rows:
        raise SystemExit("還沒有任何重新標註的結果")

    per_image_csv = OUT_DIR / "agreement.csv"
    with open(per_image_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n平均 Dice;模型為最終 U-Net 在交叉驗證中的預測(沒看過該影像)")
    print("p 為「模型 vs 新標註」與「模型 vs 舊標註」的逐張成對比較;面積比 = 新標註面積 / 舊標註面積(中位數)")
    print("無法辨識 = 重新標註時判斷看不出來而沒有畫的張數;這些影像不列入 Dice 的計算")
    print(f"{'class':<6}{'batch':<8}{'n':>4}{'無法辨識':>8}{'新 vs 舊':>10}{'模型 vs 舊':>12}{'模型 vs 新':>12}{'差異':>9}{'p':>11}{'面積比':>9}")
    for cls in NAMES.values():
        for batch in ("first", "second", "all"):
            group = [r for r in rows if r["class"] == cls and (batch == "all" or r["batch"] == batch)]
            items = [r for r in group if r["new_area"] > 0]
            unclear = len(group) - len(items)
            if not items:
                continue
            mean = lambda k: statistics.mean(r[k] for r in items)
            diffs = [r["model_vs_new"] - r["model_vs_old"] for r in items]
            ratios = [r["new_area"] / r["old_area"] for r in items if r["old_area"] > 0]
            print(
                f"{cls:<6}{batch:<8}{len(items):>4}{unclear:>8}{mean('old_vs_new'):>10.4f}{mean('model_vs_old'):>12.4f}{mean('model_vs_new'):>12.4f}"
                f"{statistics.mean(diffs):>+9.4f}{wilcoxon_p(diffs) or '-':>11}{statistics.median(ratios) if ratios else float('nan'):>9.2f}"
            )

    print("\n新舊標註最不一致的影像(各類別前 5 張)")
    for cls in NAMES.values():
        worst = sorted((r for r in rows if r["class"] == cls and r["new_area"] > 0), key=lambda r: r["old_vs_new"])[:5]
        print(f"  {cls}: " + ", ".join(f"{r['filename']}({r['batch']}, {r['old_vs_new']:.2f})" for r in worst))
    print(f"\n逐張結果: {per_image_csv}")


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("sample", help="抽樣並複製影像")
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("-n", type=int, default=N_SAMPLES)
    p.add_argument("--seed", type=int, default=SAMPLE_SEED)
    p.add_argument("--batch", default="both", choices=["both", "all", "first", "second"], help="both = 兩批各抽一半")
    p.add_argument("--force", action="store_true", help="重新抽樣並覆寫 sample.csv")
    commands.add_parser("score", help="計算新舊標註與模型預測的一致性")
    args = parser.parse_args()

    {"sample": sample, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
