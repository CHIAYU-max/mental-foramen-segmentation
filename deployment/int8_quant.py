import argparse
import csv
import platform
import random
import shutil
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

from stage2.compare_yolo_unet import GT_DIR, IOU, NAMES, ROOT, load_gt
from stage2.eval_kfold_variants import MIN_CONF
from stage2.kfold_pano import PROJECT, SUMMARY_KEYS, yolo_dir
from stage2.kfold_patient import UNET_TAG
from stage2.kfold_unet_variants import CONFIGS, build_model
from analysis.paper_metrics import image_metrics
from stage2.rerank_candidates import get_candidates

# INT8 量化(OpenVINO NNCF,2026-10-26 準備,在筆電上跑): 量化後 CPU 上的延遲與準確度變化。
#   模型: kfold_pano 第 0 折的 U-Net R34(r34_imagenet_aug)與 YOLO(預設 YOLOv8n-seg),與 pipeline_timing.py 部署用的相同。
#   校正: 第 0 折訓練集隨機 --calib 張(split seed 0),不碰評分用的影像。
#     U-Net: nncf.quantize(預設 PERFORMANCE preset),前處理與 pipeline_timing.OpenVINOUNet 相同。
#     YOLO: ultralytics export(int8=True, split="train"),內部同樣用 NNCF;FP32 與 INT8 都是 dynamic=True(與 pipeline_timing 相同)。
#       另外直接對 FP32 的 OpenVINO 模型做 nncf.quantize,讓部分分割頭保留 FP32(YOLO_FP32_SCOPES,2026-10-26 加):
#       preset MIXED、前處理(letterbox 到 imgsz、RGB、/255)與 ultralytics 的 INT8 匯出相同,只差 ignored_scope。
#       int8_ac: nncf.quantize_with_accuracy_control,依驗證指標把最敏感的層退回 FP32,直到掉分 ≤ --ac-drop。
#         驗證集用 cv(第 0 折驗證集),指標 = 每張 (mf Dice + ian Dice) / 2 的平均(後處理與評分同 YoloOV / image_metrics);
#         所以 int8_ac 的 cv 數字被拿來挑過層,只有 test 是乾淨的。
#   評分: 第 0 折驗證集(cv,112 張)與測試集(test,137 張,已排除 EXCLUDE_TEST),FP32 與 INT8 都用 OpenVINO CPU 推論,
#     指標為 paper_metrics.image_metrics;另外記錄 mf 命中由 FP32 → 各 INT8 版本改變的張數。
#   存檔一律 compress_to_fp16=False(OpenVINO 預設會把浮點權重存成 FP16)。
#     只用第 0 折單一模型(部署情境),所以數字不能與論文主表的五折多數決直接比較。
#   延遲: 測試集逐張計時(含前後處理,不含讀檔),去掉前 --warmup 張,報中位數與平均。
# 切分檔裡是伺服器的絕對路徑,在筆電上自動換成本機的 ROOT。
SERVER_ROOT = "/path/to/server/YOLOv8"
SPLIT_DIR = ROOT / "compare_results" / PROJECT / "yaml"
OUT_DIR = ROOT / "compare_results" / "int8" / platform.node()
YOLO_IMGSZ = 1280
# 保留 FP32 的範圍(OpenVINO 節點名稱的 regex;YOLOv8 的 Segment 頭是 model.22)
#   headfp32: 整個 Segment 頭(框 cv2、類別 cv3、mask 係數 cv4、prototype proto 與後處理)
#   maskfp32: 只有 mask 分支(cv4、proto),框與類別分支仍是 INT8
YOLO_FP32_SCOPES = {"int8_headfp32": r".*model\.22[./].*", "int8_maskfp32": r".*model\.22\.(cv4|proto)\..*"}


def local(line):
    return ROOT / Path(line).relative_to(SERVER_ROOT) if line.startswith(SERVER_ROOT) else Path(line)


def read_split(name):
    return [local(line) for line in (SPLIT_DIR / f"{name}.txt").read_text().splitlines() if line]


def eval_sets(limit=None, fold=0):
    # (名稱, [(stem, 影像路徑, GT 資料夾)])
    cv = [(p.stem, p, p.parent.parent / "labels") for p in read_split(f"fold{fold}_val")]
    test = [(p.stem, p, ROOT / GT_DIR) for p in read_split("test")]
    return [("cv", cv[:limit]), ("test", test[:limit])]


def calib_paths(n, fold=0):
    paths = read_split(f"fold{fold}_train")
    random.Random(0).shuffle(paths)
    return paths[:n]


class UNetOV:
    # 前後處理與 pipeline_timing.OpenVINOUNet、kfold_unet_variants.predict_masks 相同
    def __init__(self, xml, device):
        import openvino as ov
        self.size = CONFIGS[UNET_TAG]["size"]
        self.compiled = ov.Core().compile_model(str(xml), device)

    def __call__(self, path):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        start = time.perf_counter()
        h, w = image.shape
        logits = torch.from_numpy(self.compiled(unet_input(image, self.size))[0])
        pred = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False).argmax(1)[0].numpy()
        return {cls: pred == c + 1 for c, cls in NAMES.items()}, (time.perf_counter() - start) * 1000


def unet_input(image, size):
    return cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_AREA).astype(np.float32)[None, None] / 255


def export_unet(out, n_calib, fold=0):
    import nncf
    import openvino as ov
    fp32, int8 = out / "unet_fp32.xml", out / "unet_int8.xml"
    if fp32.exists() and int8.exists():
        return fp32, int8
    size = CONFIGS[UNET_TAG]["size"]
    unet = build_model({**CONFIGS[UNET_TAG], "pretrained": False})
    unet.load_state_dict(torch.load(ROOT / "compare_results" / PROJECT / "weights" / f"{UNET_TAG}_fold{fold}.pth", map_location="cpu")["model_state_dict"])
    model = ov.convert_model(unet.eval(), example_input=torch.zeros(1, 1, *size))
    ov.save_model(model, fp32, compress_to_fp16=False)
    paths = calib_paths(n_calib, fold)
    dataset = nncf.Dataset(paths, lambda p: unet_input(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), size))
    ov.save_model(nncf.quantize(model, dataset, subset_size=len(paths)), int8, compress_to_fp16=False)
    return fp32, int8


class YoloOV:
    # 推論規則與 kfold_pano.predict_fold 相同(各類別取信心最高的候選)
    def __init__(self, exported, device):
        self.model = YOLO(exported, task="segment")
        self.extra = {"device": f"intel:{device.lower()}", "imgsz": YOLO_IMGSZ}

    def __call__(self, path):
        image = cv2.imread(str(path))
        start = time.perf_counter()
        result = self.model.predict(image, conf=MIN_CONF, iou=IOU, max_det=50, retina_masks=True, verbose=False, **self.extra)[0]
        shape = tuple(result.orig_shape)
        masks = {cls: items[0][1] if items else np.zeros(shape, dtype=bool) for cls, items in get_candidates(result, shape).items()}
        return masks, (time.perf_counter() - start) * 1000


def export_yolo(tag, out, n_calib, ac_items, device, ac_drop, fold=0, variants=("int8", *YOLO_FP32_SCOPES, "int8_ac")):
    weights = yolo_dir(PROJECT, tag, fold) / "weights" / "best.pt"
    # FP32 與 pipeline_timing.py 共用;INT8 依校正設定而不同,匯出後搬到輸出資料夾(試跑與正式跑才不會混用)
    fp32 = weights.parent / "best_openvino_model"
    if not fp32.exists():
        YOLO(weights).export(format="openvino", imgsz=YOLO_IMGSZ, dynamic=True)
    models = {"fp32": fp32}
    for precision in variants:
        if precision == "int8_ac":
            precision = ac_name(ac_drop)
        dest = out / f"{tag}_{precision}_openvino_model"
        if precision == "int8":
            models[precision] = export_yolo_ultralytics(weights, dest, out, n_calib, fold)
        elif precision.startswith("int8_ac"):
            models[precision] = quantize_yolo_ac(fp32, dest, n_calib, ac_items, device, ac_drop, fold)
        else:
            models[precision] = quantize_yolo_partial(fp32, dest, YOLO_FP32_SCOPES[precision], n_calib, fold)
    return models


def export_yolo_ultralytics(weights, dest, out, n_calib, fold):
    if dest.exists():
        return dest
    # 校正用的資料集設定: train = 該折訓練集的 --calib 張(本機路徑),類別與 fold<k>.yaml 相同
    listing = out / f"calib_{n_calib}.txt"
    listing.write_text("\n".join(str(p) for p in calib_paths(n_calib, fold)) + "\n")
    names = yaml.safe_load((SPLIT_DIR / f"fold{fold}.yaml").read_text())["names"]
    data = out / f"calib_{n_calib}.yaml"
    data.write_text(yaml.safe_dump({"path": str(ROOT), "train": str(listing), "val": str(listing), "names": names}, allow_unicode=True))
    Path(YOLO(weights).export(format="openvino", imgsz=YOLO_IMGSZ, dynamic=True, int8=True, data=str(data), split="train")).rename(dest)
    return dest


def yolo_input(path):
    # 與 ultralytics INT8 匯出的校正前處理相同: letterbox 到 imgsz、BGR → RGB、CHW、/255
    from ultralytics.data.augment import LetterBox
    image = LetterBox((YOLO_IMGSZ, YOLO_IMGSZ))(image=cv2.imread(str(path)))
    return np.ascontiguousarray(image[..., ::-1].transpose(2, 0, 1))[None].astype(np.float32) / 255


def save_yolo_ov(model, fp32, dest):
    # metadata.yaml 照抄 FP32 的,YOLO() 才能直接載入
    import openvino as ov
    dest.mkdir()
    ov.save_model(model, dest / "best.xml", compress_to_fp16=False)
    shutil.copy(fp32 / "metadata.yaml", dest)
    return dest


def make_static(src, dest):
    # 動態尺寸的 YOLO OpenVINO 模型 → 固定 1×3×imgsz×imgsz;metadata 的 dynamic 改成 false,ultralytics 才會 letterbox 成正方形
    import openvino as ov
    if dest.exists():
        return dest
    model = ov.Core().read_model(next(src.glob("*.xml")))
    model.reshape([1, 3, YOLO_IMGSZ, YOLO_IMGSZ])
    dest.mkdir(parents=True)
    ov.save_model(model, dest / "best.xml", compress_to_fp16=False)
    metadata = yaml.safe_load((src / "metadata.yaml").read_text(encoding="utf-8"))
    metadata["args"]["dynamic"] = False
    (dest / "metadata.yaml").write_text(yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return dest


def quantize_yolo_partial(fp32, dest, pattern, n_calib, fold=0):
    # FP32 的 OpenVINO 模型 → INT8,符合 pattern 的節點保留 FP32
    import nncf
    import openvino as ov
    if dest.exists():
        return dest
    paths = calib_paths(n_calib, fold)
    model = ov.Core().read_model(next(fp32.glob("*.xml")))
    quantized = nncf.quantize(model, nncf.Dataset(paths, yolo_input), preset=nncf.QuantizationPreset.MIXED, subset_size=len(paths),
                              ignored_scope=nncf.IgnoredScope(patterns=[pattern]))
    return save_yolo_ov(quantized, fp32, dest)


def ac_name(max_drop):
    # accuracy control 版本的名稱: 預設門檻 0.01 → int8_ac(沿用既有匯出),其他 → int8_ac005 等
    return "int8_ac" if max_drop == 0.01 else "int8_ac" + f"{max_drop:g}".replace("0.", "")


def ac_score(predictor, item):
    # accuracy control 的逐張指標: (mf Dice + ian Dice) / 2,沒有 GT 的類別不算
    stem, path, gt_dir = item
    masks, _ = predictor(path)
    metrics = image_metrics(masks, load_gt(stem, masks["mf"].shape, gt_dir, {}))
    values = [metrics[k] for k in ("mf_dice", "ian_dice") if metrics[k] == metrics[k]]
    return float(np.mean(values)) if values else 0.0


def quantize_unet_ac(fp32, dest, n_calib, items, device, max_drop, fold=0):
    # U-Net 版的 accuracy control: 換掉 UNetOV 裡的編譯模型,前後處理與評分與 UNetOV 相同;preset 與 export_unet 的 PTQ 相同(預設)
    import nncf
    import openvino as ov
    if dest.exists():
        return dest
    size = CONFIGS[UNET_TAG]["size"]
    predictor = UNetOV(fp32, device)

    def validate(compiled, data):
        predictor.compiled = compiled
        values = [ac_score(predictor, item) for item in data]
        return float(np.mean(values)), values

    def transform(path):
        return unet_input(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), size)

    paths = calib_paths(n_calib, fold)
    quantized = nncf.quantize_with_accuracy_control(
        ov.Core().read_model(fp32), nncf.Dataset(paths, transform), nncf.Dataset(items, lambda item: transform(item[1])), validate,
        max_drop=max_drop, drop_type=nncf.DropType.ABSOLUTE, subset_size=len(paths))
    ov.save_model(quantized, dest, compress_to_fp16=False)
    return dest


def quantize_yolo_ac(fp32, dest, n_calib, items, device, max_drop, fold=0):
    # quantize_with_accuracy_control: NNCF 傳進來的是編譯好的模型,換掉 ultralytics 後端裡的模型,後處理與評分就和 YoloOV 完全相同
    import nncf
    import openvino as ov
    if dest.exists():
        return dest
    predictor = YoloOV(fp32, device)
    predictor(items[0][1])  # 先跑一次,建立 ultralytics 的 predictor
    backend = predictor.model.predictor.model.backend

    def validate(compiled, data):
        backend.ov_compiled_model = compiled
        values = [ac_score(predictor, item) for item in data]
        return float(np.mean(values)), values

    paths = calib_paths(n_calib, fold)
    model = ov.Core().read_model(next(fp32.glob("*.xml")))
    quantized = nncf.quantize_with_accuracy_control(
        model, nncf.Dataset(paths, yolo_input), nncf.Dataset(items, lambda item: yolo_input(item[1])), validate,
        max_drop=max_drop, drop_type=nncf.DropType.ABSOLUTE, preset=nncf.QuantizationPreset.MIXED, subset_size=len(paths))
    return save_yolo_ov(quantized, fp32, dest)


def run(predict, items):
    rows = []
    for stem, path, gt_dir in items:
        masks, ms = predict(path)
        gt = load_gt(stem, masks["mf"].shape, gt_dir, {})
        rows.append({"stem": stem, **image_metrics(masks, gt), "ms": ms})
    return rows


def summarize(tag, dataset, results, warmup):
    # results: {precision: rows},第一個是 fp32;其他版本另記 mf 命中相對 fp32 改變的張數
    def mean(rows, key):
        values = [r[key] for r in rows if r[key] == r[key]]
        return round(statistics.mean(values), 4) if values else float("nan")

    out = []
    base = results["fp32"]
    for precision, rows in results.items():
        ms = [r["ms"] for r in rows[warmup:]] or [r["ms"] for r in rows]
        row = {"model": tag, "set": dataset, "precision": precision, "n": len(rows), **{k: mean(rows, k) for k in SUMMARY_KEYS},
               "ms_median": round(statistics.median(ms), 1), "ms_mean": round(statistics.mean(ms), 1)}
        if precision != "fp32":
            hits = [(a["mf_hit_1R"], b["mf_hit_1R"]) for a, b in zip(base, rows)]
            row["hit_gained"] = sum(1 for a, b in hits if not a and b)
            row["hit_lost"] = sum(1 for a, b in hits if a and not b)
        out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser(description="kfold_pano 第 0 折模型的 OpenVINO INT8 量化:CPU 延遲與準確度")
    parser.add_argument("--models", nargs="+", default=["unet", "yolov8n-seg"], help="unet 和 / 或 YOLO 的 tag(例如 yolov8n-seg、yolo26s-seg)")
    parser.add_argument("--calib", type=int, default=300, help="校正張數(第 0 折訓練集)")
    parser.add_argument("--device", default="CPU", choices=["CPU", "GPU"], help="OpenVINO 裝置(GPU = Intel 內顯)")
    parser.add_argument("--ac-drop", type=float, default=0.01, help="int8_ac 容許的掉分(cv 上 (mf Dice + ian Dice) / 2,絕對值)")
    parser.add_argument("--static", action="store_true", help="YOLO 改用固定 imgsz×imgsz 輸入(內顯跑動態尺寸的 INT8 很慢),CSV 加 _static")
    parser.add_argument("--warmup",type=int, default=5, help="延遲統計去掉前幾張")
    parser.add_argument("--check", action="store_true", help="試跑: 校正 8 張、每個集合只評 5 張,輸出到 <主機>_check")
    args = parser.parse_args()

    out = OUT_DIR.with_name(OUT_DIR.name + "_check") if args.check else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    n_calib = 8 if args.check else args.calib
    sets = eval_sets(5 if args.check else None)
    summary, per_image = [], []
    for tag in args.models:
        if tag == "unet":
            fp32, int8 = export_unet(out, n_calib)
            models = {"fp32": UNetOV(fp32, args.device), "int8": UNetOV(int8, args.device)}
        else:
            exported = export_yolo(tag, out, n_calib, dict(sets)["cv"], args.device, args.ac_drop)
            if args.static:
                exported = {precision: make_static(path, out / "static" / f"{tag}_{precision}_openvino_model") for precision, path in exported.items()}
            models ={precision: YoloOV(path, args.device) for precision, path in exported.items()}
        for dataset, items in sets:
            results = {precision: run(m, items) for precision, m in models.items()}
            summary += summarize(tag, dataset, results, args.warmup)
            for precision, rs in results.items():
                per_image += [{"model": tag, "set": dataset, "precision": precision, **r} for r in rs]
            print(*(f"{r['model']} {r['set']} {r['precision']}: " + " ".join(f"{k}={r[k]}" for k in (*SUMMARY_KEYS, "ms_median")) for r in summary[-len(results):]), sep="\n", flush=True)

    # 匯出的模型與裝置無關,共用同一個資料夾;CSV 依裝置與 --static 分開
    suffix = ("" if args.device == "CPU" else f"_{args.device.lower()}") + ("_static" if args.static else "")
    write_csv(out / f"summary{suffix}.csv", summary)
    write_csv(out / f"per_image{suffix}.csv", per_image)
    print(f"輸出: {out}")


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)), restval="")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
