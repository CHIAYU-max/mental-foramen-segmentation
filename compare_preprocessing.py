import argparse
import csv
import gc
import random
import statistics
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

# 每種前處理對應的資料集資料夾(底下要有 train/images 與 valid/images)
# 請把 None 換成實際路徑,相對路徑以這個腳本所在資料夾為準;保持 None 的會被跳過
VARIANTS = {
    "original": "./datasets/20260912/mf_559_original",
    "gamma": "./datasets/20260912/mf_397_gamma",
    "bilateral_filter": "./datasets/20260913/mf_397_bilateral_filter",
    "bf_gamma": "./datasets/20260913/mf_397_bf_gamma",
    "gamma_clahe": "./datasets/20260913/mf_397_gamma_clahe",
    "bf_gamma_clahe": "./datasets/20260913/mf_397_bf_gamma_clahe",
    "gamma_clahe_unsharp_mask": "./datasets/20260913/mf_397_gamma_clahe_unsharp_mask",
    "bf_gamma_clahe_um": "./datasets/20260913/mf_397_bf_gamma_clahe_um",
}

NAMES = {0: "mf", 1: "ian"}
SEEDS = [0]
MODEL = "yolov8n-seg.pt"
PROJECT = "compare_mf_397_5fold"
# 所有 variant 共用同一份 K 折切分:依檔名(不含副檔名)決定每張影像屬於哪一折
K_FOLDS = 5
SPLIT_SEED = 0
BASELINE = "original"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TRAIN_ARGS = dict(epochs=200, patience=50, imgsz=1280, batch=16, device=0, workers=8)
# 驗證時的批次大小;GPU 記憶體不足時可用 --eval-batch 調小
EVAL = {"batch": TRAIN_ARGS["batch"]}

OUT_DIR = ROOT / "compare_results" / PROJECT
RUNS_CSV = OUT_DIR / "runs.csv"
SUMMARY_CSV = OUT_DIR / "summary.csv"

MAP_METRICS = ["box_map50", "box_map50_95", "mask_map50", "mask_map50_95"]
COUNT_METRICS = ["tp", "fp", "fn"]
PAIRED_METRICS = ["box_map50_95", "mask_map50_95"]
CLASS_METRICS = ["box_p", "box_r", "box_map50", "box_map50_95", "mask_p", "mask_r", "mask_map50", "mask_map50_95"]
FOCUS_CLASS = "mf"
FIELDS = (
    ["variant", "fold", "seed", "best_epoch", "epochs_run"]
    + ["box_p", "box_r", "box_map50", "box_map50_95"]
    + ["mask_p", "mask_r", "mask_map50", "mask_map50_95"]
    + [f"{m}_{c}" for c in NAMES.values() for m in CLASS_METRICS + COUNT_METRICS]
    + ["tp", "fp", "fn", "save_dir"]
)


def collect_images(dataset_dir):
    # 收集資料夾下所有 images/ 內的影像(已切分或未切分皆可),回傳 {檔名: 路徑}
    images = {}
    for p in sorted(dataset_dir.rglob("*")):
        if p.suffix.lower() not in IMAGE_SUFFIXES or p.parent.name != "images":
            continue
        if p.stem in images:
            raise SystemExit(f"{dataset_dir}: 檔名重複 {p.stem}\n  {images[p.stem]}\n  {p}")
        images[p.stem] = p
    if not images:
        raise SystemExit(f"{dataset_dir}: 找不到任何 images/ 資料夾內的影像")
    return images


def build_folds(all_images):
    # 所有 variant 的檔名集合必須完全相同,才能套用同一份切分
    ref_variant, ref_images = next(iter(all_images.items()))
    ref_stems = set(ref_images)
    for variant, images in all_images.items():
        stems = set(images)
        if stems != ref_stems:
            missing = sorted(ref_stems - stems)[:5]
            extra = sorted(stems - ref_stems)[:5]
            raise SystemExit(
                f"{variant} 與 {ref_variant} 的影像檔名不一致,無法共用切分\n"
                f"  {ref_variant}: {len(ref_stems)} 張, {variant}: {len(stems)} 張\n"
                f"  {variant} 缺少(前5): {missing}\n  {variant} 多出(前5): {extra}"
            )

    stems = sorted(ref_stems)
    random.Random(SPLIT_SEED).shuffle(stems)
    folds = [set(stems[i::K_FOLDS]) for i in range(K_FOLDS)]
    with open(OUT_DIR / "folds.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stem", "fold"])
        writer.writerows(sorted((s, i) for i, fold in enumerate(folds) for s in fold))
    sizes = ", ".join(f"fold{i}={len(fold)}" for i, fold in enumerate(folds))
    print(f"{K_FOLDS} 折切分: 共 {len(stems)} 張, 各折驗證集 {sizes} (split seed={SPLIT_SEED})")
    return folds


def write_data_yaml(variant, images, fold, val_stems):
    # 每個 variant、每一折各寫一份 yaml 與影像清單
    yaml_dir = OUT_DIR / "yaml"
    yaml_dir.mkdir(parents=True, exist_ok=True)
    train_txt = yaml_dir / f"{variant}_fold{fold}_train.txt"
    val_txt = yaml_dir / f"{variant}_fold{fold}_val.txt"
    train_txt.write_text("".join(f"{p}\n" for s, p in sorted(images.items()) if s not in val_stems))
    val_txt.write_text("".join(f"{p}\n" for s, p in sorted(images.items()) if s in val_stems))

    path = yaml_dir / f"{variant}_fold{fold}.yaml"
    data = {"train": str(train_txt), "val": str(val_txt), "names": NAMES}
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return path


def read_runs():
    if not RUNS_CSV.exists():
        return []
    with open(RUNS_CSV, newline="") as f:
        return list(csv.DictReader(f))


def append_run(row):
    new_file = not RUNS_CSV.exists()
    with open(RUNS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def best_epoch_info(save_dir):
    # best.pt 不保留 epoch,改用 results.csv 中 Box+Mask mAP50-95 最高的 epoch 估計
    with open(Path(save_dir) / "results.csv", newline="") as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]) + float(r["metrics/mAP50-95(M)"]))
    return int(float(best["epoch"])), len(rows)


def evaluate(variant, data_yaml, fold, seed, save_dir):
    run_name = f"{variant}_fold{fold}_seed{seed}"
    save_dir = Path(save_dir)

    best = YOLO(save_dir / "weights" / "best.pt")
    # plots 必須開啟,否則驗證器不會累計混淆矩陣(TP/FP/FN 會全為 0)
    metrics = best.val(
        data=str(data_yaml),
        imgsz=TRAIN_ARGS["imgsz"],
        batch=EVAL["batch"],
        device=TRAIN_ARGS["device"],
        workers=TRAIN_ARGS["workers"],
        split="val",
        plots=True,
        project=PROJECT,
        name=f"{run_name}_val",
    )

    best_epoch, epochs_run = best_epoch_info(save_dir)
    row = {
        "variant": variant,
        "fold": fold,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "box_p": metrics.box.mp,
        "box_r": metrics.box.mr,
        "box_map50": metrics.box.map50,
        "box_map50_95": metrics.box.map,
        "mask_p": metrics.seg.mp,
        "mask_r": metrics.seg.mr,
        "mask_map50": metrics.seg.map50,
        "mask_map50_95": metrics.seg.map,
        "save_dir": str(save_dir),
    }

    # 混淆矩陣: matrix[預測, 實際],最後一列/欄是 background
    matrix = metrics.confusion_matrix.matrix
    for i, cls in NAMES.items():
        # 各類別的 P/R/mAP50/mAP50-95;class_result 的索引是該類別在 ap_class_index 中的位置
        for prefix, metric in (("box", metrics.box), ("mask", metrics.seg)):
            class_ids = list(metric.ap_class_index)
            values = metric.class_result(class_ids.index(i)) if i in class_ids else (0.0, 0.0, 0.0, 0.0)
            for name, value in zip(("p", "r", "map50", "map50_95"), values):
                row[f"{prefix}_{name}_{cls}"] = value
        row[f"tp_{cls}"] = int(matrix[i, i])
        row[f"fp_{cls}"] = int(matrix[i, :].sum() - matrix[i, i])
        row[f"fn_{cls}"] = int(matrix[:, i].sum() - matrix[i, i])
    for m in COUNT_METRICS:
        row[m] = sum(row[f"{m}_{cls}"] for cls in NAMES.values())

    row = {k: (round(float(v), 5) if isinstance(v, float) or hasattr(v, "item") else v) for k, v in row.items()}

    del best
    gc.collect()
    torch.cuda.empty_cache()
    return row


def train_and_eval(variant, data_yaml, fold, seed):
    run_name = f"{variant}_fold{fold}_seed{seed}"

    model = YOLO(MODEL)
    model.train(data=str(data_yaml), project=PROJECT, name=run_name, seed=seed, save=True, plots=True, **TRAIN_ARGS)
    save_dir = Path(model.trainer.save_dir)
    del model

    append_run(evaluate(variant, data_yaml, fold, seed, save_dir))


def recount():
    # 用各 run 已存的 best.pt 重新驗證並改寫 runs.csv(不重新訓練);訓練進行中請勿執行
    rows = read_runs()
    new_rows = []
    for r in rows:
        data_yaml = OUT_DIR / "yaml" / f"{r['variant']}_fold{r['fold']}.yaml"
        new_rows.append(evaluate(r["variant"], data_yaml, int(r["fold"]), int(r["seed"]), r["save_dir"]))
    with open(RUNS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(new_rows)


def summarize():
    rows = read_runs()
    if not rows:
        print("尚無結果")
        return

    by_variant = {}
    for r in rows:
        by_variant.setdefault(r["variant"], {})[(r["fold"], r["seed"])] = r

    summary = []
    for variant, runs in by_variant.items():
        n_seeds = len({seed for _, seed in runs})
        item = {"variant": variant, "n": len(runs)}
        for m in MAP_METRICS:
            values = [float(r[m]) for r in runs.values()]
            item[f"{m}_mean"] = round(statistics.mean(values), 4)
            item[f"{m}_std"] = round(statistics.stdev(values), 4) if len(values) > 1 else 0.0
        # TP/FP/FN 把各折加總(全部影像各驗證一次),多個 seed 時取平均
        for m in COUNT_METRICS:
            item[f"{m}_total"] = round(sum(float(r[m]) for r in runs.values()) / n_seeds, 1)
        # 與 baseline 在相同 (fold, seed) 上的成對差異
        base = by_variant.get(BASELINE, {})
        keys = sorted(set(runs) & set(base))
        for m in PAIRED_METRICS:
            diffs = [float(runs[k][m]) - float(base[k][m]) for k in keys]
            item[f"{m}_diff_mean"] = round(statistics.mean(diffs), 4) if diffs else ""
            item[f"{m}_diff_std"] = round(statistics.stdev(diffs), 4) if len(diffs) > 1 else ""
            item[f"{m}_wins"] = f"{sum(d > 0 for d in diffs)}/{len(diffs)}"
        summary.append(item)
    summary.sort(key=lambda x: x["mask_map50_95_mean"] + x["box_map50_95_mean"], reverse=True)

    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print("\n各折平均 ± 標準差;TP/FP/FN 為各折加總")
    print(f"{'variant':<28}{'n':>3}" + "".join(f"{m:>20}" for m in MAP_METRICS) + "".join(f"{m:>8}" for m in COUNT_METRICS))
    for item in summary:
        cells = "".join(f"{item[f'{m}_mean']:>12.4f} ±{item[f'{m}_std']:<6.4f}" for m in MAP_METRICS)
        counts = "".join(f"{item[f'{m}_total']:>8}" for m in COUNT_METRICS)
        print(f"{item['variant']:<28}{item['n']:>3}{cells}{counts}")

    print(f"\n與 {BASELINE} 的成對差異(同一折、同一 seed 相減;wins = 贏過 {BASELINE} 的次數)")
    print(f"{'variant':<28}" + "".join(f"{m + ' diff':>24}{'wins':>7}" for m in PAIRED_METRICS))
    for item in summary:
        if item["variant"] == BASELINE:
            continue
        cells = ""
        for m in PAIRED_METRICS:
            mean, std = item[f"{m}_diff_mean"], item[f"{m}_diff_std"]
            text = f"{mean:+.4f} ±{std:.4f}" if std != "" else (f"{mean:+.4f}" if mean != "" else "-")
            cells += f"{text:>24}{item[f'{m}_wins']:>7}"
        print(f"{item['variant']:<28}{cells}")
    summarize_class(by_variant, FOCUS_CLASS)
    print(f"\n逐次結果: {RUNS_CSV}\n彙總結果: {SUMMARY_CSV}")


def summarize_class(by_variant, cls):
    # 單一類別的彙總;舊版 runs.csv 沒有這些欄位時需先執行 --recount
    keys = [f"{m}_{cls}" for m in CLASS_METRICS]
    first = next(iter(next(iter(by_variant.values())).values()))
    if any(k not in first or first[k] in ("", None) for k in keys):
        print(f"\n(runs.csv 缺少 {cls} 的分類別欄位,請先執行 --recount)")
        return

    table = []
    for variant, runs in by_variant.items():
        n_seeds = len({seed for _, seed in runs})
        item = {"variant": variant, "n": len(runs)}
        for k in keys:
            values = [float(r[k]) for r in runs.values()]
            item[f"{k}_mean"] = round(statistics.mean(values), 4)
            item[f"{k}_std"] = round(statistics.stdev(values), 4) if len(values) > 1 else 0.0
        for m in COUNT_METRICS:
            item[f"{m}_{cls}_total"] = round(sum(float(r[f"{m}_{cls}"]) for r in runs.values()) / n_seeds, 1)
        table.append(item)
    table.sort(key=lambda x: x[f"mask_map50_95_{cls}_mean"] + x[f"box_map50_95_{cls}_mean"], reverse=True)

    path = OUT_DIR / f"summary_{cls}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)

    print(f"\n只看 {cls}:各折平均 ± 標準差;TP/FP/FN 為各折加總")
    print(f"{'variant':<28}" + "".join(f"{m:>18}" for m in CLASS_METRICS) + "".join(f"{m:>7}" for m in COUNT_METRICS))
    for item in table:
        cells = "".join(f"{item[f'{k}_mean']:>10.3f} ±{item[f'{k}_std']:<6.3f}" for k in keys)
        counts = "".join(f"{item[f'{m}_{cls}_total']:>7}" for m in COUNT_METRICS)
        print(f"{item['variant']:<28}{cells}{counts}")
    print(f"{cls} 彙總結果: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(K_FOLDS)))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--summary-only", action="store_true", help="只重新彙總已有結果,不訓練")
    parser.add_argument("--check-only", action="store_true", help="只檢查檔名並產生切分,不訓練")
    parser.add_argument("--recount", action="store_true", help="用已存的 best.pt 重新驗證並改寫 runs.csv,不訓練")
    parser.add_argument("--eval-batch", type=int, default=EVAL["batch"], help="驗證時的批次大小")
    args = parser.parse_args()
    EVAL["batch"] = args.eval_batch

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.recount:
        recount()
    elif not args.summary_only:
        datasets = {}
        for variant in args.variants:
            if variant not in VARIANTS:
                parser.error(f"未知的 variant: {variant}")
            if VARIANTS[variant] is None:
                print(f"[跳過] {variant}: 尚未在 VARIANTS 填入資料集路徑")
                continue
            dataset_dir = (ROOT / VARIANTS[variant]).resolve()
            if not dataset_dir.is_dir():
                parser.error(f"{variant}: 找不到 {dataset_dir}")
            datasets[variant] = dataset_dir
        if not datasets:
            parser.error("沒有可訓練的 variant")
        for fold in args.folds:
            if not 0 <= fold < K_FOLDS:
                parser.error(f"fold 必須介於 0 到 {K_FOLDS - 1}")

        # 切分一律以全部 variant 的檔名為準,只跑部分 variant 時切分也不會變
        all_images = {v: collect_images((ROOT / d).resolve()) for v, d in VARIANTS.items() if d is not None}
        folds = build_folds(all_images)
        if args.check_only:
            return

        # 已完成的 (variant, fold, seed) 不重跑,中斷後可直接重新執行接續
        # 先跑完一整折的所有 variant 再換下一折,中途的結果也能成對比較
        done = {(r["variant"], int(r["fold"]), int(r["seed"])) for r in read_runs()}
        for seed in args.seeds:
            for fold in args.folds:
                for variant in datasets:
                    if (variant, fold, seed) in done:
                        print(f"[已完成] {variant} fold={fold} seed={seed}")
                        continue
                    data_yaml = write_data_yaml(variant, all_images[variant], fold, folds[fold])
                    train_and_eval(variant, data_yaml, fold, seed)

    summarize()


if __name__ == "__main__":
    main()
