import argparse
import csv
import platform
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from compare_yolo_unet import IOU, NAMES, ROOT
from eval_kfold_variants import MIN_CONF
from fdi_cropped import assign_fdi_indices, find_mandibular_border, get_split_roi_bounds, process_and_separate_jaws
from kfold_pano import EXCLUDE_TEST, PROJECT, ROI_SOURCES, yolo_dir
from kfold_patient import UNET_TAG
from kfold_unet_variants import CONFIGS, build_model, predict_masks
from mf_unsharp_mask import unsharp_mask
from predict_unet import DEVICE
from rerank_candidates import get_candidates, mask_distance

# 端到端延遲分解(2026-10-25): 從整張全景片到頦孔位置與距離,逐階段計時,找出整個系統的瓶頸。
#   與 roofline_profile.py 互補: roofline 只看兩個神經網路的逐層運算;這裡把 CPU 上的前處理、
#   第一階段後處理、找顎骨邊界、裁切、銳化、選候選、計算距離都算進來。
# 流程與研究用的腳本相同,但全部在記憶體中進行(fdi_cropped.py 會把 ROI 存成 jpg 再讀回,這裡不計檔案 I/O):
#   讀圖 → 第一階段 YOLO 牙齒分割(fdi_cropped.process_and_separate_jaws,conf 0.25、retina_masks)
#   → 牙齒過濾與上下顎分群 → FDI 編號 → 找下顎骨邊界 → 裁切左右 ROI
#   → unsharp mask(sigma 1.0、strength 0.5、threshold 3,與 mf_unsharp_mask.py 產生測試集的設定相同)
#   → 第二階段分割 mf / ian(U-Net 或 YOLO,推論規則與 kfold_pano.py 相同)
#   → mf 質心、mf 與 ian 的最短距離
# 影像: 測試集 ROI 的來源全景片(roi_sources.csv 中 set = test,排除 EXCLUDE_TEST),每張全景片最多兩個 ROI。
TEETH_MODEL = ROOT / "dental_project" / "train_success_replica3" / "weights" / "best.pt"
PANO_DIR = ROOT / "dental_project" / "original_pictures"
OUT_DIR = ROOT / "compare_results" / "roofline"
UNSHARP = dict(sigma=1.0, strength=0.5, threshold=3)
CUDA = DEVICE.type == "cuda" if isinstance(DEVICE, torch.device) else str(DEVICE).startswith("cuda")
# (階段, 執行位置);順序即表格順序
STAGES = [
    ("read", "CPU"),
    ("stage1_preprocess", "CPU"),
    ("stage1_inference", "GPU"),
    ("stage1_nms_masks", "CPU+GPU"),
    ("stage1_filter_kmeans", "CPU"),
    ("fdi", "CPU"),
    ("jawline", "CPU"),
    ("crop", "CPU"),
    ("unsharp_mask", "CPU"),
    ("stage2_model", "GPU"),
    ("stage2_select", "CPU"),
    ("measure", "CPU"),
]


def timed(fn, *args, **kwargs):
    if CUDA:
        torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    if CUDA:
        torch.cuda.synchronize()
    return out, (time.perf_counter() - start) * 1e3


class TimedYOLO:
    # 包住第一階段模型,讓 process_and_separate_jaws 內的 YOLO 呼叫與其後的過濾分開計時
    def __init__(self, model, **extra):
        self.model = model
        self.extra = extra  # OpenVINO 時指定 device 與 imgsz
        self.ms = 0.0
        self.speed = {}

    def __call__(self, *args, **kwargs):
        results, self.ms = timed(self.model, *args, **kwargs, **self.extra)
        self.speed = results[0].speed if results else {}
        return results


def test_panos():
    with open(ROI_SOURCES, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["set"] == "test" and r["stem"] not in EXCLUDE_TEST]
    return sorted({PANO_DIR / r["source_pano"] for r in rows})


def openvino_yolo(weights, imgsz):
    # 匯出成 OpenVINO(動態輸入尺寸,才能與 PyTorch 一樣用 rect letterbox),已匯出就直接用
    exported = weights.parent / f"{weights.stem}_openvino_model"
    if not exported.exists():
        YOLO(weights).export(format="openvino", imgsz=imgsz, dynamic=True)
    return YOLO(exported, task="segment")


class OpenVINOUNet:
    # U-Net 以 OpenVINO 推論;前後處理與 kfold_unet_variants.predict_masks 相同
    def __init__(self, unet, device):
        import openvino as ov
        size = CONFIGS[UNET_TAG]["size"]
        self.compiled = ov.Core().compile_model(ov.convert_model(unet.cpu(), example_input=torch.zeros(1, 1, *size)), device)

    def predict(self, image, size):
        h, w = image.shape
        x = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_AREA).astype(np.float32)[None, None] / 255
        logits = torch.from_numpy(self.compiled(x)[0])
        pred = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False).argmax(1)[0].numpy()
        return {cls: pred == c + 1 for c, cls in NAMES.items()}


def load_stage2(tag, openvino=None):
    if tag == "unet":
        unet = build_model({**CONFIGS[UNET_TAG], "pretrained": False}).to(DEVICE)
        unet.load_state_dict(torch.load(ROOT / "compare_results" / PROJECT / "weights" / f"{UNET_TAG}_fold0.pth", map_location=DEVICE)["model_state_dict"])
        return OpenVINOUNet(unet.eval(), openvino) if openvino else unet.eval()
    weights = yolo_dir(PROJECT, tag, 0) / "weights" / "best.pt"
    return openvino_yolo(weights, 1280) if openvino else YOLO(weights)


def stage2(tag, model, roi, extra):
    # 回傳 ({類別: bool 遮罩}, 模型時間, 挑選候選時間);extra = OpenVINO 時的 device 與 imgsz
    if tag == "unet":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if isinstance(model, OpenVINOUNet):
            masks, ms = timed(model.predict, gray, CONFIGS[UNET_TAG]["size"])
        else:
            masks, ms = timed(predict_masks, model, gray, CONFIGS[UNET_TAG]["size"])
        return masks, ms, 0.0
    result, ms = timed(lambda: model.predict(roi, conf=MIN_CONF, iou=IOU, max_det=50, retina_masks=True, verbose=False, **extra)[0])
    shape = tuple(result.orig_shape)
    candidates, select_ms = timed(get_candidates, result, shape)
    masks = {cls: items[0][1] if items else np.zeros(shape, dtype=bool) for cls, items in candidates.items()}
    return masks, ms, select_ms


def measure(masks):
    # mf 質心與 mf–ian 最短距離(像素);沒有預測時回傳 nan
    mf, ian = masks["mf"], masks["ian"]
    if not mf.any():
        return float("nan"), float("nan"), float("nan")
    m = cv2.moments(mf.astype(np.uint8), binaryImage=True)
    dist = mask_distance(mf, ian) if ian.any() else float("nan")
    return m["m10"] / m["m00"], m["m01"] / m["m00"], dist


def run_pano(path, teeth, tag, model, extra):
    t = {name: 0.0 for name, _ in STAGES}
    total_start = time.perf_counter()
    img, t["read"] = timed(cv2.imread, str(path))
    (upper, lower), jaws_ms = timed(process_and_separate_jaws, img, teeth)
    t["stage1_preprocess"] = teeth.speed.get("preprocess", 0.0)
    t["stage1_inference"] = teeth.speed.get("inference", 0.0)
    t["stage1_nms_masks"] = teeth.ms - t["stage1_preprocess"] - t["stage1_inference"]
    t["stage1_filter_kmeans"] = jaws_ms - teeth.ms
    lower, t["fdi"] = timed(assign_fdi_indices, lower)
    h, w = img.shape[:2]
    rois = []
    for side, b in get_split_roi_bounds(lower).items():
        jaw_y, ms = timed(find_mandibular_border, img, b["y_max_root"], b["x_min"], b["x_max"], b["dynamic_fallback"])
        t["jawline"] += ms
        roi, ms = timed(lambda: img[b["y_upper"]:min(jaw_y, h), b["x_min"]:min(b["x_max"], w)].copy())
        t["crop"] += ms
        if roi.size:
            rois.append((side, roi))
    found = []
    for side, roi in rois:
        um, ms = timed(unsharp_mask, roi, **UNSHARP)
        t["unsharp_mask"] += ms
        masks, model_ms, select_ms = stage2(tag, model, um, extra)
        t["stage2_model"] += model_ms
        t["stage2_select"] += select_ms
        (cx, cy, dist), ms = timed(measure, masks)
        t["measure"] += ms
        found.append(f"{side}:{'mf' if cx == cx else '-'}")
    if CUDA:
        torch.cuda.synchronize()
    wall = (time.perf_counter() - total_start) * 1e3
    return {"pano": path.name, "teeth": len(upper) + len(lower), "rois": len(rois), "found": " ".join(found), **{k: round(v, 3) for k, v in t.items()}, "sum_ms": round(sum(t.values()), 3), "wall_ms": round(wall, 3)}


def device_name(openvino):
    if openvino:
        import openvino as ov
        return f"OpenVINO {openvino}({ov.Core().get_property(openvino, 'FULL_DEVICE_NAME')})"
    if CUDA:
        return torch.cuda.get_device_name(0)
    return f"CPU({platform.processor() or platform.machine()},PyTorch {torch.get_num_threads()} 執行緒)"


def write_summary(rows, tag, out, openvino=None):
    total = statistics.mean(r["sum_ms"] for r in rows)
    lines = [
        f"# 端到端延遲分解: 第二階段 {tag}",
        "",
        f"- 全景片 {len(rows)} 張(測試集來源),ROI 共 {sum(r['rois'] for r in rows)} 個;裝置 {device_name(openvino)};主機 {platform.node()}",
        f"- 每張全景片平均 {total:.1f} ms(各階段相加),實測牆鐘時間平均 {statistics.mean(r['wall_ms'] for r in rows):.1f} ms;中位數 {statistics.median(r['wall_ms'] for r in rows):.1f} ms",
        f"- stage1_nms_masks = 第一階段 YOLO 呼叫的總時間減去 Ultralytics 回報的 preprocess 與 inference(NMS、retina_masks 放大遮罩、轉成多邊形)",
        "",
        "| 階段 | 執行位置 | 平均 ms | 中位數 ms | 佔比 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, where in STAGES:
        values = [r[name] for r in rows]
        lines.append(f"| {name} | {where} | {statistics.mean(values):.2f} | {statistics.median(values):.2f} | {statistics.mean(values) / total:.1%} |")
    gpu = sum(statistics.mean(r[n] for r in rows) for n, where in STAGES if where == "GPU")
    lines += ["", f"GPU 上的神經網路推論共佔 {gpu / total:.1%},其餘 {1 - gpu / total:.1%} 是 CPU 上的前後處理(Amdahl: 只加速模型,整體最多快 {total / (total - gpu):.2f} 倍)。"]
    (out / f"pipeline_{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2", default="unet", help="unet 或 kfold_pano 的 YOLO 名稱(例如 yolov8n-seg),權重用 fold 0")
    parser.add_argument("--warmup", type=int, default=3, help="前幾張全景片先跑一次不計時")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 張(試跑用)")
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--openvino", choices=["CPU", "GPU"], help="兩個模型都改用 OpenVINO 推論(GPU = Intel 內顯);第一次會先匯出模型")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    panos = test_panos()
    if args.limit:
        panos = panos[:args.limit]
    # 匯出的模型不帶訓練時的 imgsz,不指定會退回 640;PyTorch 權重則沿用訓練時的 imgsz(第一階段 1024、第二階段 1280)
    device = {"device": f"intel:{args.openvino.lower()}"} if args.openvino else {}
    teeth = TimedYOLO(openvino_yolo(TEETH_MODEL, 1024) if args.openvino else YOLO(TEETH_MODEL), **device, **({"imgsz": 1024} if args.openvino else {}))
    extra = {**device, "imgsz": 1280} if args.openvino else {}
    model = load_stage2(args.stage2, args.openvino)
    with torch.inference_mode():
        for path in panos[:args.warmup]:
            run_pano(path, teeth, args.stage2, model, extra)
        rows = []
        for i, path in enumerate(panos, 1):
            rows.append(run_pano(path, teeth, args.stage2, model, extra))
            print(f"[{i}/{len(panos)}] {path.name}: {rows[-1]['wall_ms']:.1f} ms, ROI {rows[-1]['rois']} ({rows[-1]['found']})")
    with open(out / f"pipeline_{args.stage2}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_summary(rows, args.stage2, out, args.openvino)


if __name__ == "__main__":
    main()
