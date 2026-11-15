"""2026-11-13 第一階段牙齒模型 INT8 的三種改善(使用者要求「三個都幫我試看看」)。

背景: 直接 PTQ(yolo_int8_export.py,切割輸出頭 model.22 已由 ultralytics 保留 FP32)讓牙位掉 2–4.5 點。
三種方法:
  1. mixed: 逐區塊敏感度(只把某一個區塊 model.i 留 FP32,看一致率回升多少),再依回升量貪婪地把區塊留 FP32,直到開發集一致率 ≥ 目標。
  2. aac:   NNCF quantize_with_accuracy_control,用同一個開發集指標,自動把敏感的量化點退回 FP32(max_drop = 1 − 目標)。
  3. qat:   第一階段原始訓練資料已不在伺服器,沒有牙齒標註;改用 FP32 模型當老師的蒸餾式 QAT(NNCF torch 假量化,
            loss = 學生與老師的輸出頭原始輸出 MSE),訓練影像 = 非測試、非開發集的全景片(只用影像)。
開發集指標(不看測試集): 非測試集、有頦孔 GT 的全景片,用 GT 頦孔 + 候選模型的參考線判牙位(center 規則),
  和 OpenVINO FP32 第一階段的判定相同的比例(一致率);另記參考線在頦孔高度的水平位移(px)。
事先定的判準: 開發集一致率 ≥ 0.99 的版本才進測試集五種子評分(deploy_pipeline.py --teeth-ov),
  測試集上對 score_mixed_union: 牙位(原始與複查後)差 ≥ −1 點、轉介率差在 ±1 點內,且筆電每張至少省 50 ms,才考慮採用。
輸出: compare_results/deploy_eval/stage1_int8/
"""
import argparse
import csv
import json
import random
import re
import shutil
import time
from copy import deepcopy

import cv2
import nncf
import numpy as np
import openvino as ov
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

from stage2.compare_yolo_unet import ROOT
from stage1.fdi_cropped import assign_fdi_indices, process_and_separate_jaws
from position.mf_position import DEDUP_PX, centroid, classify, load_mf, reference_lines
from deployment.pipeline_timing import PANO_DIR, TEETH_MODEL, openvino_yolo, test_panos

OUT = ROOT / "compare_results" / "deploy_eval" / "stage1_int8"
FP32_DIR = TEETH_MODEL.parent / f"{TEETH_MODEL.stem}_openvino_model"
CALIB = ROOT / "compare_results" / "deploy_eval" / "yolo_int8" / "calib_teeth" / "images"
META = ROOT / "compare_results" / "deploy_eval" / "yolo_int8" / "teeth_int8_openvino_model" / "metadata.yaml"
TARGET = 0.99
BLOCK = re.compile(r"__module\.model\.(\d+)[./]")


# ---------- 資料 ----------
def dev_set():
    # 非測試集、有頦孔 GT 的全景片;每個頦孔的象限用 FP32 的上下顎中線決定(在 ref 裡固定)
    test = {p.name for p in test_panos()}
    mfs = load_mf()
    return {p: v for p, v in sorted(mfs.items()) if p not in test and all(m["set"] == "train" for m in v)}


def train_images(dev):
    test = {p.name for p in test_panos()}
    return sorted(p for p in PANO_DIR.iterdir() if p.suffix.lower() in (".jpg", ".png") and p.name not in test and p.name not in dev)


def letterbox(img, auto):
    x = LetterBox((1024, 1024), auto=auto, stride=32)(image=img)
    return np.ascontiguousarray(x[..., ::-1].transpose(2, 0, 1))[None].astype(np.float32) / 255


def calib_dataset():
    items = sorted(CALIB.iterdir())
    return nncf.Dataset(items, lambda p: letterbox(cv2.imread(str(p)), auto=False))


# ---------- 第一階段 + 牙位 ----------
class Stage1:
    # 用 OpenVINO FP32 匯出載入 YOLO,之後把 backend 的 compiled model 換成候選模型(前後處理完全相同)
    def __init__(self, directory=FP32_DIR):
        self.yolo = YOLO(directory, task="segment")
        self.yolo.predict(np.zeros((512, 1024, 3), np.uint8), imgsz=1024, device="intel:cpu", verbose=False)
        self.backend = self.yolo.predictor.model.backend
        self.original = self.backend.ov_compiled_model

    def set(self, compiled=None):
        self.backend.ov_compiled_model = compiled or self.original
        self.backend.read_model = None

    def __call__(self, *args, **kwargs):
        return self.yolo(*args, imgsz=1024, device="intel:cpu", **kwargs)


def x_at(line, y):
    c, v = line[0], line[1]
    return c[0] + (y - c[1]) / v[1] * v[0]


def run_pano(model, pano, quads=None):
    img = cv2.imread(str(PANO_DIR / pano))
    _, lower = process_and_separate_jaws(img, model)
    lower = assign_fdi_indices(lower) if lower else []
    return img, lower


def sides_of(pano, mfs, lower, w):
    q4 = [t["cx"] for t in lower if t.get("quadrant") == 4]
    q3 = [t["cx"] for t in lower if t.get("quadrant") == 3]
    mid = (max(q4) + min(q3)) / 2 if q4 and q3 else w / 2
    seen, out = [], []
    for mf in mfs:
        m = centroid(mf["poly"])
        q = 4 if m[0] <= mid else 3
        if any(qq == q and np.linalg.norm(m - c) < DEDUP_PX for qq, c in seen):
            continue
        seen.append((q, m))
        out.append((mf["poly"], q))
    return out


def judge(lower, sides):
    res = []
    for poly, q in sides:
        lines = reference_lines(lower, q, {}) if lower else {}
        pos, info = classify(poly, lines, "center")
        y = centroid(poly)[1]
        res.append({"pos": str(pos) if pos else info.get("reason", "none"),
                    "x": {k: float(x_at(l, y)) for k, l in lines.items()}})
    return res


def build_ref(dev):
    path = OUT / "ref_fp32.json"
    if path.exists():
        return json.loads(path.read_text())
    s1 = Stage1()
    ref = {}
    for i, (pano, mfs) in enumerate(dev.items(), 1):
        img, lower = run_pano(s1, pano)
        sides = sides_of(pano, mfs, lower, img.shape[1])
        ref[pano] = {"sides": [[p.tolist(), q] for p, q in sides], "res": judge(lower, sides)}
        if i % 50 == 0:
            print(f"ref {i}/{len(dev)}", flush=True)
    path.write_text(json.dumps(ref))
    return ref


def score_pano(model, pano, r):
    _, lower = run_pano(model, pano)
    sides = [(np.array(p), q) for p, q in r["sides"]]
    res = judge(lower, sides)
    agree = [a["pos"] == b["pos"] for a, b in zip(res, r["res"])]
    shift = [abs(a["x"][k] - b["x"][k]) for a, b in zip(res, r["res"]) for k in b["x"] if k in a["x"]]
    return agree, shift


def evaluate(model, ref, panos=None):
    agree, shift, per = [], [], []
    for pano in panos or ref:
        a, s = score_pano(model, pano, ref[pano])
        agree += a
        shift += s
        per.append(float(np.mean(a)) if a else 1.0)
    return {"agree": float(np.mean(agree)), "n": len(agree), "changed": int(len(agree) - sum(agree)),
            "shift_mean": float(np.mean(shift)) if shift else 0.0, "shift_p95": float(np.percentile(shift, 95)) if shift else 0.0}, per


# ---------- OpenVINO 模型處理 ----------
def fp32_model():
    openvino_yolo(TEETH_MODEL, 1024)
    return ov.Core().read_model(next(FP32_DIR.glob("*.xml")))


def head_scope(model):
    ops = model.get_ordered_ops()
    sig = [op.get_friendly_name() for op in ops if op.get_type_name() == "Sigmoid"]
    head = sig[-1].split("/", 1)[0]
    return nncf.IgnoredScope(names=[op.get_friendly_name() for op in ops if op.get_type_name() == "Sigmoid"
                                    or op.get_friendly_name().startswith((f"{head}/", f"{head}.dfl"))])


def compile_(model):
    return ov.Core().compile_model(model, "CPU", {"PERFORMANCE_HINT": "LATENCY"})


NO_BC = False  # --no-bc: PTQ 關掉 fast bias correction(對照;11/13 結果: 一致率 0.956 → 0.941,bias correction 有幫助)


def advanced():
    from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
    return AdvancedQuantizationParameters(disable_bias_correction=True) if NO_BC else None


def full_ptq():
    path = OUT / ("ptq_nobc.xml" if NO_BC else "ptq_full.xml")
    if path.exists():
        return ov.Core().read_model(path)
    m = fp32_model()
    q = nncf.quantize(m, calib_dataset(), preset=nncf.QuantizationPreset.MIXED, subset_size=300, ignored_scope=head_scope(m),
                      advanced_parameters=advanced())
    ov.save_model(q, path, compress_to_fp16=False)
    return ov.Core().read_model(path)


def ptq_in_memory():
    # 2026-11-13 修正: 存檔再讀回的 PTQ 模型,權重已壓成 int8 常數 + 反量化(不再是 FakeQuantize),
    # keep_fp32 只能拿掉活化的量化點、權重仍是 INT8 → 混合精度要用記憶體中(權重仍是 FakeQuantize)的模型
    m = fp32_model()
    return nncf.quantize(m, calib_dataset(), preset=nncf.QuantizationPreset.MIXED, subset_size=300, ignored_scope=head_scope(m),
                         advanced_parameters=advanced())


def fq_blocks(model):
    # 每個 FakeQuantize 屬於哪些區塊(看它的使用者節點,權重與輸入活化都算在使用它的卷積那一層)
    out = {}
    for op in model.get_ops():
        if op.get_type_name() != "FakeQuantize":
            continue
        blocks = set()
        for t in op.output(0).get_target_inputs():
            m = BLOCK.match(t.get_node().get_friendly_name())
            if m:
                blocks.add(int(m.group(1)))
        out[op.get_friendly_name()] = blocks
    return out


_FP32_W, _FP32_B = {}, {}


def const_of(node):
    if node.get_type_name() == "Convert":
        node = node.input_value(0).get_node()
    return node.get_data().astype(np.float32) if node.get_type_name() == "Constant" else None


def bias_add(conv):
    # Convolution 後面接的 Add(bias 常數)
    for t in conv.output(0).get_target_inputs():
        n = t.get_node()
        if n.get_type_name() == "Add":
            return n
    return None


def fp32_weights():
    # 卷積 friendly name → FP32 權重、bias(numpy)
    if not _FP32_W:
        for op in fp32_model().get_ops():
            if op.get_type_name() == "Convolution":
                _FP32_W[op.get_friendly_name()] = const_of(op.input_value(1).get_node())
                add = bias_add(op)
                if add is not None:
                    _FP32_B[op.get_friendly_name()] = const_of(add.input_value(1).get_node())
    return _FP32_W


def keep_fp32(model, blocks):
    # 區塊內: 拿掉輸入活化的 FakeQuantize,並把卷積權重(NNCF 已壓成 int8 常數 + 反量化)換回 FP32 原始權重
    from openvino import opset13 as ops
    m = model.clone()
    for op in m.get_ops():
        if op.get_type_name() != "FakeQuantize":
            continue
        users = {BLOCK.match(t.get_node().get_friendly_name()) for t in op.output(0).get_target_inputs()}
        if any(u and int(u.group(1)) in blocks for u in users):
            op.output(0).replace(op.input_value(0))
    w = fp32_weights()
    for op in m.get_ops():
        if op.get_type_name() != "Convolution":
            continue
        b = BLOCK.match(op.get_friendly_name())
        if b and int(b.group(1)) in blocks:
            if op.input_value(1).get_node().get_type_name() != "Constant":
                op.input(1).replace_source_output(ops.constant(w[op.get_friendly_name()]).output(0))
            add, bias = bias_add(op), _FP32_B.get(op.get_friendly_name())
            if add is not None and bias is not None:  # bias correction 改過的 bias 換回原值
                add.input(1).replace_source_output(ops.constant(bias).output(0))
    m.validate_nodes_and_infer_types()
    return m


def save_candidate(model, name):
    d = OUT / f"{name}_openvino_model"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    ov.save_model(model, d / "best.xml", compress_to_fp16=False)
    shutil.copy(META, d / "metadata.yaml")
    return d


def bench(model, n=30):
    c = compile_(model)
    x = np.random.rand(1, 3, 544, 1024).astype(np.float32)
    for _ in range(5):
        c(x)
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        c(x)
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t))


def log(row):
    path = OUT / "results.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "variant", "agree", "n", "changed", "shift_mean", "shift_p95", "server_ms", "fp32_blocks", "note"])
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in w.fieldnames})
    print(row, flush=True)


# ---------- 方法 ----------
def cmd_ref(args):
    dev = dev_set()
    ref = build_ref(dev)
    print(f"開發集: 全景片 {len(ref)} 張、頦孔 {sum(len(r['res']) for r in ref.values())} 個;訓練影像 {len(train_images(dev))} 張")
    s1 = Stage1()
    m = fp32_model()
    log({"method": "ref", "variant": "fp32", "agree": 1.0, "n": sum(len(r["res"]) for r in ref.values()), "server_ms": bench(m)})
    old = YOLO(ROOT / "compare_results/deploy_eval/yolo_int8/teeth_int8_openvino_model", task="segment")
    r, _ = evaluate(lambda *a, **k: old(*a, imgsz=1024, device="intel:cpu", **k), ref)
    log({"method": "ref", "variant": "export_int8(yolo_int8_export)", **r})
    q = full_ptq()
    s1.set(compile_(q))
    r, _ = evaluate(s1, ref)
    log({"method": "ref", "variant": "ptq_full(head FP32)", **r, "server_ms": bench(q)})


def cmd_mixed(args):
    ref = build_ref(dev_set())
    q = ptq_in_memory()
    print("FakeQuantize:", sum(op.get_type_name() == "FakeQuantize" for op in q.get_ops()), "; 全部拿掉後:",
          sum(op.get_type_name() == "FakeQuantize" for op in keep_fp32(q, set(range(30))).get_ops()), flush=True)
    s1 = Stage1()
    s1.set(compile_(keep_fp32(q, set(range(30)))))
    r, _ = evaluate(s1, ref)
    log({"method": "check", "variant": "PTQ with all quantization removed (weights and biases restored)", **r})
    s1 = Stage1()
    blocks = sorted({b for v in fq_blocks(q).values() for b in v})
    rng = random.Random(0)
    sub = sorted(rng.sample(sorted(ref), min(args.sens_panos, len(ref))))
    s1.set(compile_(q))
    r, _ = evaluate(s1, ref)
    log({"method": "ptq" + ("_nobc" if NO_BC else ""), "variant": "all INT8 (head FP32)", **r, "server_ms": bench(q)})
    if r["agree"] >= TARGET:
        save_candidate(q, "ptq_nobc" if NO_BC else "ptq")
        return
    base, _ = evaluate(s1, ref, sub)
    gains = {}
    for b in blocks:
        s1.set(compile_(keep_fp32(q, {b})))
        r, _ = evaluate(s1, ref, sub)
        gains[b] = (r["agree"] - base["agree"], base["shift_mean"] - r["shift_mean"])
        log({"method": "mixed2_sens" + ("_nobc" if NO_BC else ""), "variant": f"block {b} FP32 (subset {len(sub)})", **r})
    order = sorted(blocks, key=lambda b: gains[b], reverse=True)
    kept = []
    for b in order:
        kept.append(b)
        m = keep_fp32(q, set(kept))
        s1.set(compile_(m))
        r, _ = evaluate(s1, ref)
        log({"method": "mixed2" + ("_nobc" if NO_BC else ""), "variant": f"{len(kept)} blocks FP32", **r, "server_ms": bench(m), "fp32_blocks": " ".join(map(str, kept))})
        if r["agree"] >= TARGET:
            save_candidate(m, "mixed_nobc" if NO_BC else "mixed")
            break


def cmd_aac(args):
    dev = dev_set()
    ref = build_ref(dev)
    s1 = Stage1()
    m = fp32_model()

    def validate(compiled, items):
        s1.set(compiled)
        agree, per = [], []
        for pano in items:
            a, _ = score_pano(s1, pano, ref[pano])
            agree += a
            per.append(float(np.mean(a)) if a else 1.0)
        return float(np.mean(agree)), per

    from nncf.quantization.advanced_parameters import AdvancedAccuracyRestorerParameters
    q = nncf.quantize_with_accuracy_control(
        m, calibration_dataset=calib_dataset(), validation_dataset=nncf.Dataset(sorted(ref)), validation_fn=validate,
        max_drop=1 - TARGET, drop_type=nncf.DropType.ABSOLUTE, preset=nncf.QuantizationPreset.MIXED, subset_size=300,
        ignored_scope=head_scope(m), advanced_quantization_parameters=advanced(),
        advanced_accuracy_restorer_parameters=AdvancedAccuracyRestorerParameters(ranking_subset_size=args.rank_panos, max_num_iterations=args.max_iter))
    d = save_candidate(q, "aac_nobc" if NO_BC else "aac")
    s1.set(compile_(q))
    r, _ = evaluate(s1, ref)
    n_fq = sum(op.get_type_name() == "FakeQuantize" for op in q.get_ops())
    n_full = sum(op.get_type_name() == "FakeQuantize" for op in full_ptq().get_ops())
    log({"method": "aac" + ("_nobc" if NO_BC else ""), "variant": f"accuracy control max_drop {1 - TARGET:.2f}", **r, "server_ms": bench(q),
         "note": f"FakeQuantize {n_fq} / {n_full}; {d.name}"})


def cmd_qat(args):
    dev = dev_set()
    ref = build_ref(dev)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    imgs = train_images(dev)
    device = torch.device(args.device)
    model = YOLO(TEETH_MODEL).model.float().eval()
    for p in model.parameters():
        p.requires_grad_(True)
    teacher = deepcopy(model).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    model.model[-1].dynamic = True  # anchor 每次都重算,NNCF 建圖時才看得到(否則 /stack/0 沒有統計)
    calib = nncf.Dataset(sorted(CALIB.iterdir()), lambda p: torch.from_numpy(letterbox(cv2.imread(str(p)), auto=False)))
    q = nncf.quantize(model, calib, preset=nncf.QuantizationPreset.MIXED, subset_size=300,
                      ignored_scope=nncf.IgnoredScope(patterns=[r".*model\.22\..*", r".*model/22.*", r"^/.*"], validate=False))
    q = q.to(device).eval()  # BN 維持推論統計;假量化以 STE 傳梯度
    opt = torch.optim.Adam([p for p in q.parameters() if p.requires_grad], lr=args.lr)

    def outputs(net, x):
        (_, proto), preds = net(x)
        ts = [proto] + [v for k, v in sorted(preds.items()) if torch.is_tensor(v)]
        return ts

    def batch(path, train):
        img = cv2.imread(str(path))
        if train and random.random() < 0.5:
            img = img[:, ::-1]
        if train:
            img = np.clip(img.astype(np.float32) * random.uniform(0.85, 1.15) + random.uniform(-15, 15), 0, 255).astype(np.uint8)
        return torch.from_numpy(letterbox(img, auto=True)).to(device)

    random.seed(0)
    step = 0
    hist = [float("nan")]
    if args.export_only:  # 2026-11-13: 訓練完成後 nncf.strip 需要 torchao(沒裝),改成直接載入權重再匯出
        q.load_state_dict(torch.load(OUT / f"qat_epoch{args.epochs - 1}.pt", map_location=device))
    for epoch in range(0 if args.export_only else args.epochs):
        order = imgs[:]
        random.shuffle(order)
        for i, path in enumerate(order, 1):
            x = batch(path, True)
            with torch.no_grad():
                t = outputs(teacher, x)
            s = outputs(q, x)
            loss = sum(torch.mean((a - b) ** 2) / (b.var() + 1e-6) for a, b in zip(s, t)) / len(t)
            (loss / args.accum).backward()
            hist.append(loss.item())
            if i % args.accum == 0:
                opt.step()
                opt.zero_grad()
                step += 1
            if i % 50 == 0:
                print(f"epoch {epoch} {i}/{len(order)} loss {np.mean(hist[-50:]):.5f}", flush=True)
        torch.save(q.state_dict(), OUT / f"qat_epoch{epoch}.pt")
    q = q.cpu().eval()
    head = q.model[-1] if hasattr(q, "model") else None
    if head is not None:
        head.export, head.dynamic, head.format = True, True, "openvino"
    x = torch.from_numpy(letterbox(cv2.imread(str(imgs[0])), auto=True))
    with torch.no_grad():
        m = ov.convert_model(q, example_input=x)  # 不 strip: 假量化直接轉成 OpenVINO FakeQuantize
    m.reshape({m.input(0).get_any_name(): ov.PartialShape([1, 3, -1, -1])})
    d = save_candidate(m, "qat")
    s1 = Stage1()
    s1.set(compile_(m))
    r, _ = evaluate(s1, ref)
    n_fq = sum(op.get_type_name() == "FakeQuantize" for op in m.get_ops())
    log({"method": "qat", "variant": f"distill QAT {args.epochs} epochs lr {args.lr} ({len(imgs)} images)", **r, "server_ms": bench(m),
         "note": f"FakeQuantize {n_fq}; final loss {np.mean(hist[-100:]):.5f}; {d.name}"})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["ref", "mixed", "aac", "qat"])
    p.add_argument("--sens-panos", type=int, default=100)
    p.add_argument("--rank-panos", type=int, default=60)
    p.add_argument("--max-iter", type=int, default=30)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--device", default="cpu")
    p.add_argument("--export-only", action="store_true")
    p.add_argument("--no-bc", action="store_true", help="PTQ 關掉 bias correction")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    globals()["NO_BC"] = args.no_bc
    globals()["cmd_" + args.cmd](args)


if __name__ == "__main__":
    main()
