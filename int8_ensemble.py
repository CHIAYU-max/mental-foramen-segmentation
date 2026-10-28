import argparse

import numpy as np

from compare_yolo_unet import NAMES, load_gt
from int8_quant import OUT_DIR, UNetOV, YoloOV, ac_name, eval_sets, export_unet, export_yolo, quantize_unet_ac, write_csv
from kfold_pano import SUMMARY_KEYS
from paper_metrics import image_metrics

# INT8 五折(2026-10-26,筆電): int8_quant.py 只量化第 0 折;這裡把 5 折都量化,在測試集上算「單一模型平均」與「五折逐像素多數決」,
# 規則與 kfold_pano.evaluate 的 test single / test ensemble 相同,可直接對照論文主表。
#   U-Net: fp32 / int8(PTQ)/ int8_ac*;YOLO: fp32 / int8_ac*(各折用自己的驗證集做 accuracy control,所以只報 test,不報 cv)。
#     int8_ac* 依 --ac-drop 命名: 0.01 → int8_ac,0.005 → int8_ac005(2026-10-26 加 U-Net 的 accuracy control 與多個門檻)。
#   第 0 折沿用 int8_quant.py 的匯出(compare_results/int8/<主機>/),其他折放在 fold<k>/ 子資料夾。
#   預測遮罩以 packbits 暫存在記憶體(137 張 × 5 折 × 2 精度)。
FOLDS = [0, 1, 2, 3, 4]


def fold_out(out, fold):
    path = out if fold == 0 else out / f"fold{fold}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def pack(masks):
    return {cls: (np.packbits(m), m.shape) for cls, m in masks.items()}


def unpack(packed):
    return {cls: np.unpackbits(bits, count=int(np.prod(shape))).reshape(shape).astype(bool) for cls, (bits, shape) in packed.items()}


def fold_models(tag, out, fold, n_calib, args):
    # 每個 --ac-drop 各一個 accuracy control 版本(已匯出的直接沿用);U-Net 另有 PTQ 的 int8
    cv = dict(eval_sets(5 if args.check else None, fold))["cv"]
    if tag == "unet":
        fp32, int8 = export_unet(out, n_calib, fold)
        paths = {"fp32": fp32, "int8": int8}
        if not args.no_unet_ac:
            paths |= {ac_name(d): quantize_unet_ac(fp32, out / f"unet_{ac_name(d)}.xml", n_calib, cv, args.device, d, fold) for d in args.ac_drop}
        return {precision: UNetOV(path, args.device) for precision, path in paths.items()}
    paths = {}
    for d in args.ac_drop:
        paths |= export_yolo(tag, out, n_calib, cv, args.device, d, fold, variants=("int8_ac",))
    return {precision: YoloOV(path, args.device) for precision, path in paths.items()}


def main():
    parser = argparse.ArgumentParser(description="kfold_pano 五折模型的 OpenVINO INT8 量化:測試集單一模型平均與五折多數決")
    parser.add_argument("--models", nargs="+", default=["unet", "yolov8n-seg"])
    parser.add_argument("--calib", type=int, default=300, help="校正張數(各折訓練集)")
    parser.add_argument("--device", default="CPU", choices=["CPU", "GPU"])
    parser.add_argument("--ac-drop", type=float, nargs="+", default=[0.01], help="accuracy control 的門檻,可給多個(例如 0.01 0.005)")
    parser.add_argument("--no-unet-ac", action="store_true", help="U-Net 只做 PTQ,不做 accuracy control")
    parser.add_argument("--check", action="store_true", help="試跑: 第 0、1 折、校正 8 張、測試集 5 張,輸出到 <主機>_check")
    args = parser.parse_args()

    out = OUT_DIR.with_name(OUT_DIR.name + "_check") if args.check else OUT_DIR
    folds = FOLDS[:2] if args.check else FOLDS
    n_calib = 8 if args.check else args.calib
    test = dict(eval_sets(5 if args.check else None))["test"]
    summary, per_image = [], []
    for tag in args.models:
        preds = {}  # {precision: {fold: {stem: packed masks}}}
        for fold in folds:
            for precision, model in fold_models(tag, fold_out(out, fold), fold, n_calib, args).items():
                preds.setdefault(precision, {})[fold] = {stem: pack(model(path)[0]) for stem, path, _ in test}
            print(f"{tag} fold{fold} 完成", flush=True)
        rows = {}  # {precision: {"single": [...], "ensemble": [...]}}
        for stem, _, gt_dir in test:
            gt = None
            for precision, by_fold in preds.items():
                masks = [unpack(by_fold[fold][stem]) for fold in folds]
                gt = gt if gt is not None else load_gt(stem, masks[0]["mf"].shape, gt_dir, {})
                per_model = [image_metrics(m, gt) for m in masks]
                single = {k: float(np.nanmean([m.get(k, np.nan) for m in per_model])) for k in per_model[0]}
                vote = {cls: np.sum([m[cls] for m in masks], axis=0) > len(folds) / 2 for cls in NAMES.values()}
                ensemble = image_metrics(vote, gt)
                rows.setdefault(precision, {"single": [], "ensemble": []})
                rows[precision]["single"].append(single)
                rows[precision]["ensemble"].append(ensemble)
                per_image.append({"model": tag, "precision": precision, "stem": stem, **{f"ens_{k}": v for k, v in ensemble.items()}})
        for setting in ("single", "ensemble"):
            base = rows["fp32"][setting]
            for precision, by_setting in rows.items():
                rs = by_setting[setting]
                row = {"model": tag, "setting": f"test {setting}", "precision": precision, "folds": " ".join(map(str, folds)), "n": len(rs),
                       **{k: round(float(np.nanmean([r[k] for r in rs])), 4) for k in SUMMARY_KEYS}}
                if setting == "ensemble" and precision != "fp32":
                    hits = [(a["mf_hit_1R"], b["mf_hit_1R"]) for a, b in zip(base, rs)]
                    row["hit_gained"] = sum(1 for a, b in hits if not a and b)
                    row["hit_lost"] = sum(1 for a, b in hits if a and not b)
                summary.append(row)
                print(f"{tag} {row['setting']} {precision}: " + " ".join(f"{k}={row[k]}" for k in SUMMARY_KEYS), flush=True)

    suffix = "" if args.ac_drop == [0.01] and args.no_unet_ac else "_ac" + "_".join(ac_name(d)[7:] or "01" for d in args.ac_drop)  # 第一版(只有 0.01、U-Net 無 ac)的檔名不變
    write_csv(out / f"ensemble_summary{suffix}.csv", summary)
    write_csv(out / f"ensemble_per_image{suffix}.csv", per_image)
    print(f"輸出: {out}")


if __name__ == "__main__":
    main()
