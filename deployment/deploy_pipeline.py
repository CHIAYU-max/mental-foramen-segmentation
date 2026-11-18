"""2026-11-13 完整部署流程的整合評估(EXPERIMENTS.md 待辦「完整部署流程的整合評估」,使用者要求提前做)。

一支程式從全景片跑到「牙位 + 要不要轉介」,全部即時計算(不讀 items.json、per_side.csv 等存好的中間結果):
  讀圖 → 第一階段牙齒分割(train_success_replica3,imgsz 1024)+ FDI 編號 → 左右 ROI(stage1_eval.auto_rois 的同一套裁切)
  → 存讀 JPEG → unsharp mask → 存讀 JPEG(部署要重現兩次 JPEG,見「自動 ROI 五種子重算」)
  → 第二階段 U-Net R34/512 OpenVINO INT8 PTQ(int8_pano/models/s<種子>_fold<k>_int8.xml),串接式:
     先第 0、1 折,頦孔 IoU ≥ 0.5(或兩者都沒有頦孔)就用交集,否則補第 2–4 折做五折多數決(同 pipeline_timing --policy cascade)
  → 頦孔只留最大區塊(lcc)→ 牙位: v2 參考線(mf_position.reference_lines,由這次的第一階段結果即時算)+ 中央一半
  → 缺牙 / FDI 錯旗標(11/4 定案): 32 類 Tufts 牙齒模型與現行編號在 x4–x6 任一不一致(fdi_flag.side_status),
     或 頦孔側比對側少 ≥ 1 顆(side_count_flag);任一成立 / 沒有牙位 → 轉介。
模式:
  --score(伺服器 CPU,種子 0–4): 對測試集評分。第一階段、參考線、旗標每張全景片只算一次,第二階段每個種子各跑一次。
  --time(筆電,種子 0): 逐段計時,前 --warmup 張先跑一次不計時;不需要 GT。
輸出: compare_results/deploy_eval/<name>/(per_side.csv、timing.csv、summary.md)。
"""
import argparse
import csv
import os
import platform
import statistics
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from stage2.compare_yolo_unet import NAMES, ROOT
from analysis.e2e_eval import jpeg, largest_contour
from stage1.fdi_cropped import assign_fdi_indices, find_mandibular_border, get_split_roi_bounds, process_and_separate_jaws
from position.fdi_flag import side_status
from deployment.int8_quant import unet_input
from stage2.kfold_patient import UNET_TAG
from stage2.kfold_unet_variants import CONFIGS
from position.mf_position import centroid, classify, contour, reference_lines
from stage2.mf_postproc import largest
from data.mf_unsharp_mask import unsharp_mask
from analysis.pano_uncertainty import iou
from deployment.pipeline_timing import PANO_DIR, TEETH_MODEL, UNSHARP, openvino_yolo, test_panos, vote
from stage1.stage1_eval import Sized
from stage1.teeth32_compare import WEIGHTS as TEETH32, new_teeth

OUT = ROOT / "compare_results" / "deploy_eval"
INT8 = ROOT / "compare_results" / "int8_pano" / "models"
SIZE = CONFIGS[UNET_TAG]["size"]
TAU = 0.5
QUADRANT = {"Right": 4, "Left": 3}
STAGES = ["read", "stage1", "crop", "preprocess", "stage2_model", "stage2_post", "position", "flags"]


class Clock:
    def __init__(self):
        self.t = defaultdict(float)

    def __call__(self, name, fn, *args, **kwargs):
        start = time.perf_counter()
        out = fn(*args, **kwargs)
        self.t[name] += (time.perf_counter() - start) * 1e3
        return out


class Int8UNet:
    def __init__(self, xml, threads, size=None):
        import openvino as ov
        config = {"INFERENCE_NUM_THREADS": threads} if threads else {}
        self.compiled = ov.Core().compile_model(xml if isinstance(xml, ov.Model) else str(xml), "CPU", config)
        self.size = size or SIZE

    def __call__(self, gray):
        h, w = gray.shape
        logits = torch.from_numpy(self.compiled(unet_input(gray, self.size))[0])
        up = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        if FAST_ARGMAX:
            # 2026-11-14 --fast-argmax: torch 的 argmax 在 ROI 解析度上每次約 18–42 ms(unet_call_split.py);三類直接比大小,
            #   平手時與 argmax 相同(取編號較小的類別),輸出逐像素相同
            l0, l1, l2 = up[0].numpy()
            return {NAMES[0]: (l1 > l0) & (l1 >= l2), NAMES[1]: (l2 > l0) & (l2 > l1)}
        pred = up.argmax(1)[0].numpy()
        return {cls: pred == c + 1 for c, cls in NAMES.items()}


class TorchUNet:
    # 2026-11-15 --device cuda: 同一組權重(與 INT8 版同一個種子)以 PyTorch 在 GPU 上跑,FP32 或 FP16(--precision);
    #   放大回 ROI 與快速 argmax 也在 GPU 上做,最後才搬回 CPU(.cpu() 同步,計時包含 GPU 實際運算時間)
    def __init__(self, weights, size, half):
        from stage2.kfold_unet_variants import build_model
        net = build_model({**CONFIGS[UNET_TAG], "pretrained": False})
        net.load_state_dict(torch.load(weights, map_location="cpu")["model_state_dict"])
        self.net, self.half, self.size = net.eval().cuda(), half, size
        if half:
            self.net.half()

    def __call__(self, gray):
        h, w = gray.shape
        x = torch.from_numpy(np.ascontiguousarray(unet_input(gray, self.size))).cuda()
        logits = self.net(x.half() if self.half else x).float()
        up = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        if FAST_ARGMAX:
            l0, l1, l2 = up[0]
            return {NAMES[0]: ((l1 > l0) & (l1 >= l2)).cpu().numpy(), NAMES[1]: ((l2 > l0) & (l2 > l1)).cpu().numpy()}
        pred = up.argmax(1)[0].cpu().numpy()
        return {cls: pred == c + 1 for c, cls in NAMES.items()}


THROTTLE_OFF = False  # 2026-11-16: main() 預設呼叫 no_throttle.disable()(--allow-throttle 可關)
DEVICE = "cpu"  # 2026-11-15 --device: cpu = OpenVINO(正式部署版本);cuda = 三個模型都用 PyTorch 在 NVIDIA GPU 上跑


class Device:
    # OpenVINO 匯出的 YOLO 要指定 device;PyTorch 時 extra 為空
    def __init__(self, model, **extra):
        self.model, self.extra = model, extra

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs, **self.extra)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs, **self.extra)


class Pre:
    # 2026-11-14: --pre 推論前的防禦性前處理(robust_defense.py)
    def __init__(self, model, fn):
        self.model, self.fn = model, fn

    def __call__(self, gray):
        return self.model(self.fn(gray))


def cascade(models, gray, clock):
    # 2026-11-14: FIRST(--first-tag)= 混合式串接,先跑兩個便宜的模型(第 0、1 折),一致就停;有分歧改跑 models 五折多數決(便宜模型的輸出不用)
    preds = [clock("stage2_model", m, gray) for m in (FIRST or models[:2])]
    a, b = preds[0]["mf"], preds[1]["mf"]
    extra = 0
    if not (a.any() == b.any() and (not a.any() or iou(a, b) >= TAU)):
        if FIRST:
            extra, preds = 2, [clock("stage2_model", m, gray) for m in models]
        else:
            preds += [clock("stage2_model", m, gray) for m in models[2:]]
    # 2026-11-13: --cascade-out union = 兩折一致就停時取聯集(預設交集;11/1 FP32 串接式的聯集補回 Dice)
    merge = (lambda ps: {c: ps[0][c] | ps[1][c] for c in ps[0]}) if CASCADE_OUT == "union" and len(preds) == 2 else vote
    out = clock("stage2_post", lambda: (lambda v: {**v, "mf": largest(v["mf"])})(merge(preds)))
    return out, len(preds) + extra


FIRST, FIRST_TAG, PRE = None, None, None  # --first-tag / --pre
FLIP_FLAG = False  # --flip-flag
FAST_ARGMAX = True  # 2026-11-14 使用者確認設為預設(輸出逐位元相同);--no-fast-argmax 可關
MARGIN_REFER = 0.138  # --margin-refer: 邊界距離(個半寬)小於此值就轉介;0 = 關。2026-11-14 使用者確認加入正式部署版本(門檻 = 開發集預測頦孔 margin 的第 10 百分位,referral_combo.py)
CASCADE_OUT = "intersection"
YOLO_INT8 = OUT / "yolo_int8"
UNET_WEIGHTS = None  # 2026-11-13: --unet-tag 第二階段改用另一個設定的權重(例如 r34_imagenet_aug_slimhalf),INT8 現做並快取
# 2026-11-18 使用者確認: 正式部署版本(命令列)的第二階段改用瘦身解碼器(score_slimhalf 五種子: 牙位 +1.36 點、ian Dice −0.010;time10 每張 −144 ms)。
# 只改命令列預設;import 本模組的分析程式(UNET_WEIGHTS = None)仍是主模型 r34_imagenet_aug,和 11/18 以前的結果可比
DEPLOY_UNET_TAG = "r34_imagenet_aug_slimhalf"
TEETH_IMGSZ, T32_IMGSZ = 1024, 1024  # --teeth-imgsz / --t32-imgsz
TEETH_OV = None  # 2026-11-13: --teeth-ov 指定第一階段的 OpenVINO 模型資料夾(stage1_int8.py 的候選),32 類維持 INT8


class Imgsz:
    # new_teeth 固定傳 imgsz=1024,這裡改成指定的解析度
    def __init__(self, model, imgsz):
        self.model, self.imgsz = model, imgsz

    def predict(self, *args, **kwargs):
        kwargs["imgsz"] = self.imgsz
        return self.model.predict(*args, **kwargs)


def load_yolo(kind):
    teeth, t32 = _load_yolo(kind)
    teeth.imgsz = TEETH_IMGSZ
    return teeth, (Imgsz(t32, T32_IMGSZ) if T32_IMGSZ != 1024 else t32)


def _load_yolo(kind):
    # pt = PyTorch(伺服器評分的原設定);ov_fp32 = OpenVINO FP32 動態輸入(筆電計時的原設定);ov_int8 = yolo_int8_export.py 的 INT8
    if DEVICE == "cuda":  # 2026-11-15: 兩個牙齒模型都是 PyTorch FP32 在 GPU 上(32 類模型沒有 INT8;解析度照 --t32-imgsz)
        return Sized(Device(YOLO(TEETH_MODEL), device=0), 1024), Device(YOLO(TEETH32), device=0)
    if kind == "pt":
        return Sized(YOLO(TEETH_MODEL), 1024), YOLO(TEETH32)
    if TEETH_OV:
        teeth = YOLO(TEETH_OV, task="segment")
        t32 = YOLO(YOLO_INT8 / "teeth32_int8_openvino_model", task="segment")
        return Sized(Device(teeth, device="intel:cpu"), 1024), Device(t32, device="intel:cpu")
    if kind == "ov_fp32":
        teeth, t32 = openvino_yolo(TEETH_MODEL, 1024), openvino_yolo(TEETH32, 1024)
    elif kind == "ov_mixed":  # 2026-11-13: 第一階段 FP32(牙位靠它的牙齒輪廓),只有旗標用的 32 類模型 INT8
        teeth = openvino_yolo(TEETH_MODEL, 1024)
        t32 = YOLO(YOLO_INT8 / "teeth32_int8_openvino_model", task="segment")
    else:
        teeth = YOLO(YOLO_INT8 / "teeth_int8_openvino_model", task="segment")
        t32 = YOLO(YOLO_INT8 / "teeth32_int8_openvino_model", task="segment")
    return Sized(Device(teeth, device="intel:cpu"), 1024), Device(t32, device="intel:cpu")


def stage1(img, teeth, model32, clock):
    # 回傳 (lower, rois{side: box}, sides{quadrant: (lines, flag_fdi, flag_side, notes)})
    _, lower = clock("stage1", process_and_separate_jaws, img, teeth)
    lower = clock("stage1", assign_fdi_indices, lower) if lower else []
    h, w = img.shape[:2]
    rois = {}
    def crop():
        for side, b in get_split_roi_bounds(lower).items():
            jaw_y = min(find_mandibular_border(img, b["y_max_root"], b["x_min"], b["x_max"], b["dynamic_fallback"]), h)
            x2 = min(b["x_max"], w)
            if jaw_y > b["y_upper"] and x2 > b["x_min"]:
                rois[side] = (b["x_min"], b["y_upper"], x2, jaw_y)
    if lower:
        clock("crop", crop)
    sides = {}
    new = clock("flags", new_teeth, model32, img)
    count = {q: sum(1 for t in lower if t.get("fdi_id", "")[:1] == q and t["fdi_id"][1:] in "1234567") for q in "34"}
    for q in (3, 4):
        notes = {}
        lines = clock("position", reference_lines, lower, q, notes) if lower else {}
        def flags():
            cur = [(t["fdi_id"], contour(t)) for t in lower if t.get("quadrant") == q and len(t.get("fdi_id", "")) == 2]
            st = side_status(cur, new, str(q))
            other = "3" if q == 4 else "4"
            return int(any(st[n] != "same" for n in "456")), int(count[other] - count[str(q)] >= 1), st
        flag_fdi, flag_side, st = clock("flags", flags)
        sides[q] = (lines, flag_fdi, flag_side, st)
    if FLIP_FLAG:
        # 2026-11-14 --flip-flag: 全景片水平翻轉再跑一次第一階段,side_rows 用它的參考線再判一次牙位,兩次不同就轉介(stage1_flag_robust.py 的 flipP)
        def flip():
            _, lf = process_and_separate_jaws(np.ascontiguousarray(img[:, ::-1]), teeth)
            lf = assign_fdi_indices(lf) if lf else []
            return {q: (reference_lines(lf, {3: 4, 4: 3}[q], {}) if lf else {}) for q in (3, 4)}
        sides["flip"], sides["w"] = clock("stage1", flip), w
    return lower, rois, sides


def side_rows(img, rois, sides, models, clock):
    # 每個有 ROI 的側: 第二階段 + 牙位;回傳 {quadrant: row}, {quadrant: (box, masks)}
    rows, masks = {}, {}
    for side, (x1, y1, x2, y2) in rois.items():
        q = QUADRANT[side]
        roi = clock("preprocess", lambda: jpeg(unsharp_mask(jpeg(img[y1:y2, x1:x2]), **UNSHARP)))
        gray = clock("preprocess", cv2.cvtColor, roi, cv2.COLOR_BGR2GRAY)
        pred, n = cascade(models, gray, clock)
        lines, flag_fdi, flag_side, st = sides[q]
        def position():
            poly = largest_contour(pred["mf"])
            if poly is None or not lines:
                return ""
            return str(classify(poly + (x1, y1), lines, "center")[0] or "")
        pos = clock("position", position)
        # 2026-11-14: 預測頦孔質心到最近判定邊界(|d| = 0.5 × 半寬)的距離 / 半寬(同 position_seeds.margin),給「邊界距離轉介」用
        def boundary_margin():
            poly = largest_contour(pred["mf"]) + (x1, y1)
            m = centroid(poly)
            vals = []
            for c, _, nrm in lines.values():
                half = float(np.ptp((poly - m) @ nrm)) / 2
                vals.append(abs(abs(float((m - c) @ nrm)) - 0.5 * half) / max(half, 1e-6))
            return round(min(vals), 4)
        margin = clock("position", boundary_margin) if pos else ""
        flag_flip = 0
        if "flip" in sides and pos:
            def flipped():
                poly = largest_contour(pred["mf"]) + (x1, y1)
                poly = np.column_stack([sides["w"] - 1 - poly[:, 0], poly[:, 1]])
                lf = sides["flip"][q]
                return int(not lf or str(classify(poly, lf, "center")[0] or "") != pos)
            flag_flip = clock("position", flipped)
        rows[q] = {"side": side, "quadrant": q, "models_run": n, "mf_found": int(pred["mf"].any()), "pred": pos,
                   "flag_fdi": flag_fdi, "flag_side": flag_side, "flag_flip": flag_flip, "margin": margin,
                   "flag_margin": int(margin != "" and margin < MARGIN_REFER),
                   "refer": int(pos == "" or flag_fdi or flag_side or flag_flip or (margin != "" and margin < MARGIN_REFER)),
                   **{f"x{k}": st[k] for k in "456"}}
        masks[q] = ((x1, y1, x2, y2), pred)
    return rows, masks


def load_stage2(seed, threads, precision="int8"):
    # 第二階段的模型: 五折 +(--first-tag)先跑的兩個便宜模型 +(--pre)前處理
    global FIRST
    models = load_models(seed, threads, precision)
    FIRST = load_models(seed, threads, precision, FIRST_TAG, (0, 1)) if FIRST_TAG else None
    if PRE:
        med = lambda im: cv2.medianBlur(im, 3)
        if PRE == "median3":
            fns = [med] * 5
        else:  # median3_zscore: 再線性調成該折訓練影像的平均 / 標準差
            from stage2.quality_methods import train_stats, zscore
            stats = [train_stats(k)[1:] for k in range(5)]
            fns = [lambda im, k=k: zscore(med(im), *stats[k]) for k in range(5)]
        models = [Pre(m, fns[k]) for k, m in enumerate(models)]
        FIRST = FIRST and [Pre(m, fns[k]) for k, m in enumerate(FIRST)]
    return models


def load_models(seed, threads, precision="int8", tag=None, folds=range(5)):
    # 2026-11-13: --precision fp32 = 同一組權重直接轉 OpenVINO FP32(同 int8_seeds 的 ov_fp32 對照),用來在同一台機器比 INT8 快多少
    tag = tag or UNET_WEIGHTS or UNET_TAG
    size = CONFIGS[tag]["size"]
    if DEVICE == "cuda":
        from stage2.kfold_pano import PROJECT
        if tag != UNET_TAG or precision not in ("fp32", "fp16"):
            raise SystemExit("--device cuda 只支援主模型(r34_imagenet_aug)與 --precision fp32 / fp16")
        suffix = f"_s{seed}" if seed else ""
        def path(k):
            # 主結果種子 1–4 的原始 FP32 權重已刪,只剩 11/10 重訓的 <PROJECT>_rt(和 CPU 的 INT8 模型不是同一組權重;種子 0 相同)
            p = ROOT / "compare_results" / PROJECT / "weights" / f"{tag}{suffix}_fold{k}.pth"
            return p if p.exists() else ROOT / "compare_results" / f"{PROJECT}_rt" / "weights" / p.name
        return [TorchUNet(path(k), size, precision == "fp16") for k in folds]
    if precision == "int8" and tag == UNET_TAG:
        return [Int8UNet(INT8 / f"s{seed}_fold{k}_int8.xml", threads) for k in range(5)]
    import openvino as ov
    from stage2.kfold_pano import PROJECT
    from stage2.kfold_unet_variants import build_model
    suffix = f"_s{seed}" if seed else ""
    models = []
    for k in folds:
        xml = INT8 / f"{tag}_s{seed}_fold{k}_int8.xml"
        if precision == "int8" and xml.exists():
            models.append(Int8UNet(xml, threads, size))
            continue
        unet = build_model({**CONFIGS[tag], "pretrained": False})
        weights = ROOT / "compare_results" / PROJECT / "weights" / f"{tag}{suffix}_fold{k}.pth"
        if not weights.exists():  # 種子 1–4 重訓的權重在 <PROJECT>_rt(run_slimhalf_rt.sh)
            weights = ROOT / "compare_results" / f"{PROJECT}_rt" / "weights" / weights.name
        unet.load_state_dict(torch.load(weights, map_location="cpu")["model_state_dict"])
        model = ov.convert_model(unet.eval(), example_input=torch.zeros(1, 1, *size))
        if precision == "int8":  # 與 int8_seeds.one 相同: nncf.quantize 預設 preset、各折訓練集 300 張校正
            import nncf
            from deployment.int8_quant import calib_paths
            paths = calib_paths(300, k)
            model = nncf.quantize(model, nncf.Dataset(paths, lambda q: unet_input(cv2.imread(str(q), cv2.IMREAD_GRAYSCALE), size)), subset_size=len(paths))
            ov.save_model(model, xml, compress_to_fp16=False)
            model = xml
        models.append(Int8UNet(model, threads, size))
    return models


OVERLAYS = {"ded574b5-45a0-4f42-8737-46345c09c238": "最佳效能", "961cc777-2547-4f9d-8174-7d86181b8a7a": "最佳省電",
            "00000000-0000-0000-0000-000000000000": "平衡", "3af9b8d9-7c97-431d-ad78-34a8bfea439f": "更好的效能"}


def environment():
    # 2026-11-14(使用者同意): 計時前記錄電源狀態、電量、電源模式與當下的 CPU 使用率;11/14 下午曾在電池供電時量到每張 3.8 s(正常約 0.9 s)
    # 2026-11-17: 改用 timing_env.snapshot(),另外記可用記憶體、分頁檔用量與 OMEN 模式(--omen-mode)
    import deployment.timing_env as timing_env
    return timing_env.snapshot()


def run_time(args, out):
    env = environment()
    env["throttle_off"] = int(THROTTLE_OFF)
    print(f"環境: 電源 {env['power']}、電量 {env['battery']}%{'(充電中)' if env['charging'] else ''}、電源模式 {env['mode']}、開始前 CPU 使用率 {env['cpu_load']}%、"
          f"power throttling {'已關閉' if THROTTLE_OFF else '沒有關閉(背景執行可能變慢)'}", flush=True)
    if env["power"] == "battery" and not args.allow_battery:
        raise SystemExit("筆電沒有接電源(電池供電時每張會慢到約 4 倍),不計時。接上電源後再跑;真的要量請加 --allow-battery。")
    # 2026-11-15(使用者要求): 充電中也拒絕;11/14 time5 那輪是接上電源後在充電,整體偏慢(crop / preprocess 是正常的 1.8 / 1.3 倍)
    if env["charging"] and not args.allow_charging:
        raise SystemExit(f"電池還在充電(電量 {env['battery']}%),充電中量到的時間偏慢,不計時。等充滿再跑;真的要量請加 --allow-charging(結果要註明)。")
    panos = sorted(p for p in Path(args.panos_dir).iterdir() if p.suffix.lower() in (".jpg", ".png")) if args.panos_dir else test_panos()
    panos = panos[:args.limit or None]
    teeth, model32 = load_yolo(args.yolo)
    models = load_stage2(0, args.threads, args.precision)
    rows, sides_out = [], []
    with torch.inference_mode():
        for path in panos[:args.warmup]:
            img = cv2.imread(str(path))
            _, rois, sides = stage1(img, teeth, model32, Clock())
            side_rows(img, rois, sides, models, Clock())
        for i, path in enumerate(panos, 1):
            clock = Clock()
            start = time.perf_counter()
            img = clock("read", cv2.imread, str(path))
            _, rois, sides = stage1(img, teeth, model32, clock)
            srows, _ = side_rows(img, rois, sides, models, clock)
            wall = (time.perf_counter() - start) * 1e3
            rows.append({"pano": path.name, "rois": len(rois), "models_run": sum(r["models_run"] for r in srows.values()),
                         **{s: round(clock.t[s], 3) for s in STAGES}, "wall_ms": round(wall, 3)})
            sides_out += [{"pano": path.name, **r} for r in srows.values()]
            print(f"[{i}/{len(panos)}] {wall:.0f} ms, ROI {len(rois)}, 模型 {rows[-1]['models_run']} 次", flush=True)
    write(out / "timing.csv", rows)
    write(out / "per_side.csv", sides_out)
    import openvino as ov
    import deployment.timing_env as timing_env
    end = environment()
    try:
        grow = int(end["pagefile_mb"]) - int(env["pagefile_mb"])
    except (TypeError, ValueError):
        grow = None
    hw = ov.Core().get_property('CPU', 'FULL_DEVICE_NAME') + (f" + {torch.cuda.get_device_name(0)}" if DEVICE == "cuda" else "")
    lines = [f"# 部署流程延遲({platform.node()},{hw})", "",
             f"- 全景片 {len(rows)} 張(暖機 {args.warmup} 張不計),ROI {sum(r['rois'] for r in rows)} 個,"
             f"每個 ROI 平均跑 {sum(r['models_run'] for r in rows) / max(1, sum(r['rois'] for r in rows)):.2f} 個 {args.precision.upper()} U-Net;"
             f"第一階段與 32 類模型 {dict(pt='PyTorch CUDA FP32' if DEVICE == 'cuda' else 'PyTorch CPU', ov_fp32='OpenVINO FP32', ov_int8='OpenVINO INT8', ov_mixed='第一階段 OpenVINO FP32、32 類 INT8')[args.yolo] if not TEETH_OV else f'第一階段 {TEETH_OV}、32 類 INT8'};串接式輸出 {CASCADE_OUT}{f';先跑 {FIRST_TAG}' if FIRST_TAG else ''}{f';前處理 {PRE}' if PRE else ''};U-Net 執行緒 {args.threads or '預設'}",
             f"- 環境: 開始 {env['time']},電源 {env['power']}、電量 {env['battery']}%{'(充電中)' if env['charging'] else ''}、電源模式 {env['mode']}、"
             f"開始前 CPU 使用率 {env['cpu_load']}%、power throttling {'已關閉' if env['throttle_off'] else '沒有關閉'};結束時電源 {end['power']}",
             f"- 記憶體: 開始 {timing_env.describe(env)};結束 {timing_env.describe(end)}"
             + (";**分頁檔增加超過 200 MB,這一輪可能有用到分頁檔,數字不能用**" if grow is not None and grow > 200 else ""),
             f"- 每張全景片牆鐘時間 平均 {statistics.mean(r['wall_ms'] for r in rows):.1f} ms、中位數 {statistics.median(r['wall_ms'] for r in rows):.1f} ms、"
             f"p90 {np.percentile([r['wall_ms'] for r in rows], 90):.1f} ms", "",
             "| 階段 | 平均 ms | 中位數 ms | 佔比 |", "| --- | --- | --- | --- |"]
    total = statistics.mean(sum(r[s] for s in STAGES) for r in rows)
    for s in STAGES:
        v = [r[s] for r in rows]
        lines.append(f"| {s} | {statistics.mean(v):.1f} | {statistics.median(v):.1f} | {statistics.mean(v) / total:.1%} |")
    text = "\n".join(lines)
    (out / "summary.md").write_text(text + "\n", encoding="utf-8")
    print(text)


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def run_score(args, out):
    from analysis.e2e_eval import gt_polygons, test_rois
    from analysis.e2e_seeds import scored
    from position.mf_position import KEY_CSV
    keys = [k for k in csv.DictReader(open(KEY_CSV, newline="")) if k["set"] == "test"]
    alias_of = {(k["pano"], int(k["quadrant"])): k["alias"] for k in keys}
    rois_gt = test_rois()
    panos = sorted({k["pano"] for k in keys} | set(rois_gt))[:args.limit or None]
    teeth, model32 = load_yolo(args.yolo)
    cache = {}
    for i, pano in enumerate(panos, 1):
        img = cv2.imread(str(PANO_DIR / pano))
        _, rois, sides = stage1(img, teeth, model32, Clock())
        cache[pano] = (rois, sides)
        if i % 20 == 0:
            print(f"第一階段 [{i}/{len(panos)}]", flush=True)
    del teeth, model32
    rows, mf_rows = [], []
    with torch.inference_mode():
        for seed in args.seeds:
            models = load_stage2(seed, args.threads, args.precision)   # 11/15: 原本沒傳 precision(CPU 預設 int8 不變;--device cuda 要 fp32 / fp16)
            for pano in panos:
                img = cv2.imread(str(PANO_DIR / pano))
                rois, sides = cache[pano]
                srows, masks = side_rows(img, rois, sides, models, Clock())
                for q in (3, 4):
                    if (pano, q) in alias_of:
                        r = srows.get(q) or {"side": "", "quadrant": q, "models_run": 0, "mf_found": 0, "pred": "",
                                             "flag_fdi": sides[q][1], "flag_side": sides[q][2], "flag_flip": 0, "margin": "", "flag_margin": 0, "refer": 1}
                        rows.append({"seed": seed, "pano": pano, "alias": alias_of[(pano, q)], **r})
                for stem, side, x, y, w, h in rois_gt.get(pano, []):
                    q = QUADRANT[side]
                    zero = {"mf_dice": 0.0, "mf_hit_1R": 0.0, "ian_dice": 0.0}
                    if q not in masks:
                        m = zero
                    else:
                        (x1, y1, x2, y2), pred = masks[q]
                        polys = gt_polygons(stem, x, y, w, h)
                        gt = {}
                        for cls in NAMES.values():
                            g = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
                            for poly in polys[cls]:
                                cv2.fillPoly(g, [np.round(poly - (x1, y1)).astype(np.int32)], 1)
                            gt[cls] = g.astype(bool)
                        m = scored(pred, gt) if gt["mf"].any() else zero
                    mf_rows.append({"model": UNET_TAG, "condition": "deploy_int8_cascade", "seed": seed, "stem": stem,
                                    **{k: m.get(k, float("nan")) for k in zero}})
            print(f"種子 {seed} 完成", flush=True)
    write(out / "per_side.csv", rows)
    write(out / "per_roi.csv", mf_rows)
    summarize_score(rows, mf_rows, out)
    if args.baseline:
        compare_baseline(rows, mf_rows, OUT / args.baseline, out)


def compare_baseline(rows, mf_rows, base_dir, out):
    # 和另一次部署評分(例如 score = PyTorch 牙齒模型 + 交集)逐項配對: 差 = 這次 − 基準
    from analysis.e2e_seeds import compare
    from position.position_seeds import boot
    import io, contextlib
    base_mf = list(csv.DictReader(open(base_dir / "per_roi.csv", newline="", encoding="utf-8")))
    seeds = {str(r["seed"]) for r in mf_rows}
    base_mf = [r for r in base_mf if str(r["seed"]) in seeds]  # 只比兩邊都有的種子
    for r in base_mf:
        r["condition"] = "baseline"
        for k in ("mf_hit_1R", "mf_dice", "ian_dice"):
            r[k] = float(r[k])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = compare(base_mf + mf_rows, (UNET_TAG, "baseline"), (UNET_TAG, "deploy_int8_cascade"))
    lines = [f"## 5. 與 {base_dir.name} 配對(差 = 這次 − 基準)"] + ["- " + s for s in buf.getvalue().strip().splitlines()]
    write(out / "vs_baseline_mf.csv", res)
    base = list(csv.DictReader(open(base_dir / "per_side.csv", newline="", encoding="utf-8")))
    if MARGIN_REFER and "flag_margin" not in base[0]:
        lines.append("- 注意: 基準是 11/14 以前的轉介規則(沒有邊界距離轉介),下面「轉介」的差包含規則改變;要比較轉介請用 score_official 當基準")
    key = lambda r: (str(r["seed"]), r["alias"])
    b = {key(r): r for r in base}
    common = [r for r in rows if key(r) in b]
    same = lambda f: sum(str(r[f]) == str(b[key(r)][f]) for r in common)
    lines.append(f"- 逐側(種子 × 側 {len(common)} 個)相同: 牙位 {same('pred')}、轉介 {same('refer')}、兩種編號旗標 {same('flag_fdi')}、"
                 f"顆數旗標 {same('flag_side')}、有 ROI {sum(bool(r['side']) == bool(b[key(r)]['side']) for r in common)}、跑的模型數 {same('models_run')}")
    for lab_name, path in (("原始標註", ROOT / "mf_position/labels_rater.csv"),
                           ("複查後標註", ROOT / "compare_results/mf_position/labels_adjudicated_dir/labels_rater.csv")):
        lab = {r["alias"]: r["position"] for r in csv.DictReader(open(path, newline="", encoding="utf-8-sig"))}
        aliases = sorted({r["alias"] for r in common if lab.get(r["alias"]) in tuple("123456")})
        acc = lambda src, a: np.mean([str(r["pred"]) == lab[a] for r in src if r["alias"] == a])
        d = {a: acc(common, a) - acc([b[key(r)] for r in common], a) for a in aliases}
        pano = {r["alias"]: r["pano"] for r in common}
        pp = defaultdict(lambda: [0.0, 0])
        for a, v in d.items():
            pp[pano[a]][0] += v
            pp[pano[a]][1] += 1
        lo, hi, p = boot(list(pp.values()))
        lines.append(f"- 牙位({lab_name},正常牙 {len(aliases)} 個): 差 {100 * np.mean(list(d.values())):+.2f} 點 [{100 * lo:+.2f}, {100 * hi:+.2f}],p {p:.2f}")
    text = "\n".join(lines)
    with open(out / "summary.md", "a", encoding="utf-8") as f:
        f.write("\n" + text + "\n")
    print(text)


def summarize_score(rows, mf_rows, out):
    from analysis.e2e_seeds import compare
    from position.position_referral import group
    from position.position_seeds import boot
    MP = ROOT / "compare_results" / "mf_position"
    lines = []
    # 1. 頦孔命中: 對照 = e2e_seeds 的人工 ROI、FP32 五折(主結果)
    base = [r for r in csv.DictReader(open(ROOT / "compare_results/e2e_seeds/per_roi_s01234.csv", newline=""))
            if r["model"] == UNET_TAG and r["condition"] in ("manual", "auto_jpeg")]
    for r in base:
        for k in ("mf_hit_1R", "mf_dice", "ian_dice"):
            r[k] = float(r[k])
    lines.append("## 1. 頦孔(測試 ROI,對種子平均,以病人為組)")
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = compare(base + mf_rows, (UNET_TAG, "manual"), (UNET_TAG, "deploy_int8_cascade"))
        res += compare(base + mf_rows, (UNET_TAG, "auto_jpeg"), (UNET_TAG, "deploy_int8_cascade"))
    lines += ["- " + s for s in buf.getvalue().strip().splitlines()]
    write(out / "summary_mf.csv", res)
    runs = [r["models_run"] for r in rows if r["models_run"]]
    lines.append(f"- 每個 ROI 平均跑 {np.mean(runs):.2f} 個模型(在兩個模型就停的比例 {np.mean([n == 2 for n in runs]):.1%})")

    # 2. 牙位(正常牙,strict),對照 = position_seeds 的主結果(人工 ROI、FP32 五折、items.json 參考線)
    orig = {r["alias"]: r for r in csv.DictReader(open(ROOT / "mf_position/labels_rater.csv", newline="", encoding="utf-8-sig"))}
    adj = {r["alias"]: r for r in csv.DictReader(open(MP / "labels_adjudicated_dir/labels_rater.csv", newline="", encoding="utf-8-sig"))}
    main = defaultdict(list)
    for r in csv.DictReader(open(MP / "position_seeds/per_seed.csv", newline="")):
        if r["variant"] == "R34/512 FP32 五折(主結果)":
            main[r["alias"]].append(r)
    seeds = sorted({r["seed"] for r in rows})
    present = {r["alias"] for r in rows}
    lines.append("\n## 2. 牙位(正常牙 1–6,strict;差 = 部署 − 主結果,以全景片 bootstrap)")
    for lab, name in ((orig, "原始標註"), (adj, "複查後標註")):
        aliases = [a for a in main if a in present and lab.get(a, {}).get("position") in tuple("123456")]
        dep = {a: np.mean([r["pred"] == lab[a]["position"] for r in rows if r["alias"] == a]) for a in aliases}
        ref = {a: np.mean([r["pred"] == lab[a]["position"] for r in main[a]]) for a in aliases}
        per_seed = [np.mean([r["pred"] == lab[r["alias"]]["position"] for r in rows if r["seed"] == s and r["alias"] in dep]) for s in seeds]
        pano = {a: main[a][0]["pano"] for a in aliases}
        pp = defaultdict(lambda: [0.0, 0])
        for a in aliases:
            pp[pano[a]][0] += dep[a] - ref[a]
            pp[pano[a]][1] += 1
        lo, hi, p = boot(list(pp.values()))
        d = np.mean([dep[a] - ref[a] for a in aliases])
        lines.append(f"- {name}(n = {len(aliases)}): 部署 {100 * np.mean(per_seed):.1f} ± {100 * np.std(per_seed, ddof=1):.1f}%"
                     f"(各種子 {' / '.join(f'{100 * x:.1f}' for x in per_seed)});主結果 {100 * np.mean(list(ref.values())):.1f}%;"
                     f"差 {100 * d:+.2f} 點 [{100 * lo:+.2f}, {100 * hi:+.2f}],p {p:.2f}")

    # 3. 系統層級(全部已標註的側,與 position_referral 的 P1 相同定義)
    labeled = sorted(a for a in main if a in present and (adj.get(a, {}).get("position") or adj.get(a, {}).get("fdi_wrong") == "1"))
    grp = {a: group(adj[a]) for a in labeled}
    lines.append(f"\n## 3. 系統層級(測試集已標註 {len(labeled)} 側,含缺牙與 FDI 錯;5 種子平均;對原始標註)")
    for lab, name in ((orig, "原始標註"), (adj, "複查後標註")):
        res = []
        for s in seeds:
            by = {r["alias"]: r for r in rows if r["seed"] == s}
            o = {}
            for a in labeled:
                r = by[a]
                o[a] = "refer" if r["refer"] else ("ok" if grp[a] == "正常牙" and r["pred"] == lab[a]["position"] else "wrong")
            res.append(o)
        ref_rate = np.mean([[o[a] == "refer" for a in labeled] for o in res])
        silent = np.mean([[o[a] == "wrong" for a in labeled] for o in res])
        acc = np.mean([sum(o[a] == "ok" for a in labeled) / max(1, sum(o[a] != "refer" for a in labeled)) for o in res])
        ab = [a for a in labeled if grp[a] != "正常牙"]
        nm = [a for a in labeled if grp[a] == "正常牙"]
        lines.append(f"- {name}: 轉介 {100 * ref_rate:.1f}%、自動回答中正確 {100 * acc:.1f}%、靜默錯誤 {100 * silent:.1f}%;"
                     f"異常側被轉介 {np.mean([[o[a] == 'refer' for a in ab] for o in res]) * len(ab):.1f} / {len(ab)}、"
                     f"正常牙被轉介 {np.mean([[o[a] == 'refer' for a in nm] for o in res]) * len(nm):.1f} / {len(nm)}")
    ref = list(csv.DictReader(open(MP / "referral/summary.csv", newline="")))
    p1 = next(r for r in ref if r["labels"] == "orig" and r["policy"].startswith("P1"))
    lines.append(f"- 對照(position_referral P1,FP32 五折、存好的旗標,原始標註): 轉介 {100 * float(p1['referred']) / int(p1['sides']):.1f}%、"
                 f"自動回答中正確 {100 * float(p1['auto_accuracy']):.1f}%、靜默錯誤 {100 * float(p1['silent_wrong']) / int(p1['sides']):.1f}%")

    # 4. 即時計算的中間結果與存好的版本一致嗎
    fdi = {r["alias"]: int(r["flag_x456"]) for r in csv.DictReader(open(MP / "fdi_flag/per_side.csv", newline=""))}
    side = {r["alias"]: int(float(r["diff"]) >= 1) for r in csv.DictReader(open(MP / "side_count/per_side.csv", newline=""))}
    s0 = {r["alias"]: r for r in rows if r["seed"] == seeds[0]}
    common = [a for a in labeled if a in s0]
    lines.append("\n## 4. 即時計算 vs 存好的中間結果(種子 0)")
    lines.append(f"- 兩種編號不一致旗標相同 {sum(s0[a]['flag_fdi'] == fdi.get(a) for a in common)} / {len(common)};"
                 f"左右顆數旗標相同 {sum(s0[a]['flag_side'] == side.get(a) for a in common)} / {len(common)};"
                 f"有自動 ROI 的側 {sum(bool(s0[a]['side']) for a in common)} / {len(common)}")
    text = "\n".join(lines)
    (out / "summary.md").write_text(text + "\n", encoding="utf-8")
    print(text)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--score", action="store_true")
    mode.add_argument("--time", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--threads", type=int, default=0, help="U-Net OpenVINO 執行緒(0 = 預設)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--panos-dir")
    parser.add_argument("--openvino-yolo", action="store_true", help="計時: 第一階段與 32 類模型也用 OpenVINO FP32 CPU")
    parser.add_argument("--name", default=None)
    parser.add_argument("--precision", choices=["int8", "fp32", "fp16"], default="int8", help="第二階段 U-Net 的精度(計時比較用;fp16 只用在 --device cuda)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                        help="cpu = OpenVINO(正式部署版本);cuda = 兩個牙齒模型 PyTorch FP32 + U-Net PyTorch FP32 / FP16 在 NVIDIA GPU 上(2026-11-15)")
    # 2026-11-13 定為正式部署版本: 第一階段 OpenVINO FP32 + 32 類 INT8(ov_mixed)、U-Net INT8 串接式聯集輸出
    parser.add_argument("--yolo", choices=["pt", "ov_fp32", "ov_int8", "ov_mixed"], default=None,
                        help="兩個牙齒模型的格式;預設 ov_mixed(正式部署版本);--openvino-yolo 為舊寫法 = ov_fp32")
    parser.add_argument("--cascade-out", choices=["intersection", "union"], default="union",
                        help="串接式只跑兩折時的輸出;預設 union(正式部署版本),11/13 上午的 score 是 intersection")
    # 2026-11-13 晚(使用者確認)正式部署版本再加: 第一階段後處理只算框內(逐像素相同)、32 類模型 imgsz 640
    parser.add_argument("--no-fast-masks", dest="fast_masks", action="store_false",
                        help="關掉 fast_masks.py(第一階段原圖解析度 mask 只算框內,逐像素相同;預設開)")
    parser.add_argument("--unet-tag", help="第二階段用這個設定的權重(kfold_unet_variants.CONFIGS);預設 r34_imagenet_aug_slimhalf(2026-11-18 起的正式部署版本),"
                        "r34_imagenet_aug = 11/18 以前的主模型;--device cuda 只支援主模型,沒指定時自動用主模型")
    parser.add_argument("--teeth-imgsz", type=int, default=1024)
    parser.add_argument("--t32-imgsz", type=int, default=640, help="32 類模型(旗標)的輸入解析度;預設 640(正式部署版本),11/13 白天的結果是 1024")
    parser.add_argument("--teeth-ov", help="第一階段改用這個 OpenVINO 模型資料夾(覆蓋 --yolo 的第一階段;32 類 INT8)")
    # 2026-11-14(使用者要求): 防禦性前處理與混合式串接,預設都關
    parser.add_argument("--pre", choices=["median3", "median3_zscore"], help="U-Net 推論前對 ROI 做 3×3 中值濾波(+ zscore)")
    parser.add_argument("--first-tag", help="混合式串接: 先跑這個設定的第 0、1 折(例如 r34_imagenet_aug_384),有分歧改跑主模型五折")
    # 2026-11-14(使用者確認)正式部署版本再加: 邊界距離轉介,預設 0.138;11/14 以前的結果是 0(關)
    parser.add_argument("--margin-refer", type=float, default=0.138, help="預測頦孔離判定邊界不到這麼多個半寬就轉介;0 = 關(11/14 以前的結果)")
    parser.add_argument("--fast-argmax", dest="fast_argmax", action="store_true", default=True, help="U-Net 輸出用直接比大小取類別(與 argmax 逐像素相同,每次呼叫省約 17 ms;預設開)")
    parser.add_argument("--no-fast-argmax", dest="fast_argmax", action="store_false", help="改回 torch.argmax(11/14 以前的做法,結果相同、較慢)")
    parser.add_argument("--allow-battery", action="store_true", help="計時: 電池供電時也照量(預設拒絕;結果不能和接電源的比較)")
    parser.add_argument("--allow-charging", action="store_true", help="計時: 充電中也照量(預設拒絕;充電中偏慢,結果要註明)")
    parser.add_argument("--flip-flag", action="store_true", help="多跑一次水平翻轉的第一階段,兩次的牙位不同就轉介(預設關;第一階段時間約 2 倍)")
    parser.add_argument("--baseline", help="評分: 另一次部署評分的資料夾名稱(compare_results/deploy_eval/ 底下),逐項配對比較")
    # 2026-11-16(使用者同意): 在背景視窗跑久了 Windows 會把行程節能排程(EcoQoS),第一階段慢約 1.5 倍;預設一開始就關掉節流
    parser.add_argument("--omen-mode", choices=["eco", "balanced", "performance", "unknown"],
                        help="計時: OMEN Gaming Hub 當時的模式(程式讀不到,手動填;記進 summary.md 與 timing_env_log.csv,2026-11-17)")
    parser.add_argument("--allow-throttle", action="store_true", help="不要關掉 Windows 的 power throttling(2026-11-16 以前的行為;背景執行時第一階段會慢約 1.5 倍)")
    args = parser.parse_args()
    global CASCADE_OUT, TEETH_OV, UNET_WEIGHTS, TEETH_IMGSZ, T32_IMGSZ, FIRST_TAG, PRE, FLIP_FLAG, MARGIN_REFER, FAST_ARGMAX, DEVICE, THROTTLE_OFF
    # 2026-11-17: 只有 --time 才寫 timing_env_log.csv(評分不算計時);OMEN 模式由 --omen-mode 傳給 timing_env
    if args.omen_mode:
        os.environ["OMEN_MODE"] = args.omen_mode
    if not args.time:
        os.environ.setdefault("TIMING_ENV_LOG", "0")
    if not args.allow_throttle:
        import deployment.no_throttle as no_throttle
        THROTTLE_OFF = no_throttle.disable()
    elif args.time:
        import deployment.timing_env as timing_env
        timing_env.begin()
    DEVICE = args.device
    if DEVICE == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda: 這台機器沒有可用的 CUDA GPU")
        if args.precision == "int8":
            args.precision = "fp32"  # GPU 上沒有 INT8 版本,預設改用 FP32
        args.yolo = "pt"
    elif args.precision == "fp16":
        raise SystemExit("--precision fp16 只用在 --device cuda")
    FAST_ARGMAX = args.fast_argmax
    FLIP_FLAG, MARGIN_REFER = args.flip_flag, args.margin_refer
    FIRST_TAG, PRE = args.first_tag, args.pre
    if args.unet_tag is None:  # 2026-11-18: 正式部署版本 = 瘦身解碼器;GPU 版還沒有瘦身解碼器,維持主模型
        args.unet_tag = UNET_TAG if DEVICE == "cuda" else DEPLOY_UNET_TAG
    UNET_WEIGHTS, TEETH_IMGSZ, T32_IMGSZ = args.unet_tag, args.teeth_imgsz, args.t32_imgsz
    CASCADE_OUT = args.cascade_out
    TEETH_OV = args.teeth_ov
    if args.fast_masks:
        import deployment.fast_masks as fast_masks
        fast_masks.install()
    if args.yolo is None:
        args.yolo = "ov_fp32" if args.openvino_yolo else "ov_mixed"
    out = OUT / (args.name or ("score" if args.score else f"time_{platform.node()}"))
    out.mkdir(parents=True, exist_ok=True)
    run_score(args, out) if args.score else run_time(args, out)


if __name__ == "__main__":
    main()
