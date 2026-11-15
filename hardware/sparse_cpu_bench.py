"""2026-11-13 2:4 結構化稀疏在 CPU(OpenVINO)上會不會變快(使用者要求;不需要病人影像)。

- 權重: 種子 0 第 0 折。dense = 主結果 r34_imagenet_aug_fold0.pth;2:4 = sp24d_ft_fold0.pth(45 層 2:4、conv1 與 head.1 dense,已微調)。
- 版本: FP32 dense、FP32 2:4、INT8 dense(int8_pano 的 PTQ 模型)、INT8 2:4(這裡用 NNCF PTQ 現做;校正用 Tufts ROI 300 張,
  因為只量速度,校正資料只影響準確度)。另對 INT8 兩個版本開 OpenVINO 的 CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE=0.5(官方文件說只用在
  有 AMX 的 CPU 上的 INT8 全連接層,預期卷積不受影響)。
- 量法同 unet_int8_bench.py: batch 1、512×512 隨機輸入,各暖機 --warmup 次,所有版本輪流 --rounds 輪 × --iters 次,報中位數。
- 輸出: compare_results/deploy_eval/sparse_bench_<主機>/(per_iter.csv、summary.md)。
"""
import argparse
import csv
import platform
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import torch

from stage2.compare_yolo_unet import ROOT
from deployment.int8_quant import unet_input
from stage2.kfold_pano import PROJECT
from stage2.kfold_patient import UNET_TAG
from stage2.kfold_unet_variants import CONFIGS, build_model

SIZE = CONFIGS[UNET_TAG]["size"]
OUT = ROOT / "compare_results" / "deploy_eval"
MODELS = ROOT / "compare_results" / "int8_pano" / "models"


def torch_model(path):
    state = torch.load(path, map_location="cpu")
    state = state.get("model_state_dict", state)
    unet = build_model({**CONFIGS[UNET_TAG], "pretrained": False})
    unet.load_state_dict(state)
    return unet.eval()


def sparsity(unet):
    # 回傳 (2:4 的卷積層數, 卷積層數, 卷積權重為 0 的比例)
    ok, n, zero, total = 0, 0, 0, 0
    for m in unet.modules():
        if isinstance(m, torch.nn.Conv2d):
            w = m.weight.detach()
            n += 1
            zero += int((w == 0).sum())
            total += w.numel()
            if w.shape[1] % 4 == 0:
                g = w.permute(0, 2, 3, 1).reshape(-1, 4)
                ok += int(((g != 0).sum(1) <= 2).all())
    return ok, n, zero / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--calib", type=int, default=300)
    parser.add_argument("--calib-dir", default=str(ROOT / "external" / "tufts" / "rois_laptop_x1.58"))
    args = parser.parse_args()
    out = OUT / f"sparse_bench_{platform.node()}"
    out.mkdir(parents=True, exist_ok=True)
    core = ov.Core()
    example = torch.zeros(1, 1, *SIZE)
    dense = torch_model(ROOT / "compare_results" / PROJECT / "weights" / f"{UNET_TAG}_fold0.pth")
    sparse = torch_model(MODELS / "sp24d_ft_fold0.pth")
    info = {"dense": sparsity(dense), "2:4": sparsity(sparse)}
    ir = {"fp32_dense": ov.convert_model(dense, example_input=example), "fp32_sp24": ov.convert_model(sparse, example_input=example)}
    sp_int8 = MODELS / "sp24d_ft_fold0_ovint8_tufts.xml"
    if not sp_int8.exists():
        import nncf
        paths = sorted(Path(args.calib_dir).glob("*.png"))
        paths = [paths[i] for i in np.random.default_rng(0).choice(len(paths), min(args.calib, len(paths)), replace=False)]
        data = nncf.Dataset(paths, lambda p: unet_input(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), SIZE))
        ov.save_model(nncf.quantize(ir["fp32_sp24"], data, subset_size=len(paths)), sp_int8, compress_to_fp16=False)
    variants = {
        "FP32 dense": (ir["fp32_dense"], {}),
        "FP32 2:4": (ir["fp32_sp24"], {}),
        "INT8 dense": (str(MODELS / "s0_fold0_int8.xml"), {}),
        "INT8 2:4": (str(sp_int8), {}),
        "INT8 dense + sparse 選項": (str(MODELS / "s0_fold0_int8.xml"), {"CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE": 0.5}),
        "INT8 2:4 + sparse 選項": (str(sp_int8), {"CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE": 0.5}),
    }
    compiled, notes = {}, {}
    for name, (model, config) in variants.items():
        try:
            compiled[name] = core.compile_model(model, "CPU", config)
        except Exception as e:  # 選項不支援時記下來,不中斷
            notes[name] = f"compile 失敗: {type(e).__name__}: {str(e).splitlines()[0][:120]}"
    x = np.random.default_rng(0).integers(0, 256, (1, 1, *SIZE)).astype(np.float32) / 255
    for m in compiled.values():
        for _ in range(args.warmup):
            m(x)
    rows = []
    names = list(compiled)
    for r in range(args.rounds):
        for name in (names if r % 2 == 0 else names[::-1]):
            for i in range(args.iters):
                start = time.perf_counter()
                compiled[name](x)
                rows.append({"round": r, "variant": name, "iter": i, "ms": (time.perf_counter() - start) * 1e3})
    with open(out / "per_iter.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    med = {n: statistics.median(r["ms"] for r in rows if r["variant"] == n) for n in names}
    rnd = {n: [statistics.median(r["ms"] for r in rows if r["variant"] == n and r["round"] == k) for k in range(args.rounds)] for n in names}
    lines = [f"# 2:4 稀疏在 CPU 上的速度({platform.node()},{core.get_property('CPU', 'FULL_DEVICE_NAME')},OpenVINO {ov.__version__})", "",
             f"- batch 1、512×512 隨機輸入;暖機各 {args.warmup} 次,輪流 {args.rounds} 輪 × {args.iters} 次",
             f"- 權重檢查: dense 卷積 0 的比例 {info['dense'][2]:.1%};2:4 模型 {info['2:4'][0]} / {info['2:4'][1]} 層符合 2:4,0 的比例 {info['2:4'][2]:.1%}",
             f"- INT8 2:4 = NNCF PTQ,用 Tufts ROI {args.calib} 張校正(只量速度)", "",
             "| 版本 | 中位數 ms | 各輪中位數 | 相對 FP32 dense |", "| --- | --- | --- | --- |"]
    base = med.get("FP32 dense")
    for n in names:
        lines.append(f"| {n} | {med[n]:.1f} | {' / '.join(f'{v:.1f}' for v in rnd[n])} | {base / med[n]:.2f}× |")
    if "INT8 dense" in med and "INT8 2:4" in med:
        lines.append(f"\n- 2:4 相對 dense: FP32 {med['FP32 dense'] / med['FP32 2:4']:.2f}×、INT8 {med['INT8 dense'] / med['INT8 2:4']:.2f}×")
    lines += [f"- {k}: {v}" for k, v in notes.items()]
    text = "\n".join(lines)
    (out / "summary.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
