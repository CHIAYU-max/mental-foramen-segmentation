#!/usr/bin/env python3
"""
Layer-wise roofline profiling for the mental-foramen segmentation models.

Run on the GPU server, for example:
  python roofline_profile.py --model unet --out roofline_unet
  python roofline_profile.py --model unet --weights compare_results/kfold_pano_mf_559/weights/r34_imagenet_aug_fold0.pth --out roofline_unet
  python roofline_profile.py --model yolo --weights runs/segment/train/weights/best.pt --out roofline_yolo
  python roofline_profile.py --model yolo --fuse --out roofline_yolo_fused

FLOPs and bytes depend only on the architecture and input size, not on the
trained weights, so --weights is optional.

Outputs written to --out:
  layers.csv     one row per executed leaf module (in execution order)
  by_type.csv    time / FLOPs / bytes grouped by operator type
  aten_ops.txt   torch.profiler table of the slowest low-level ops
                 (catches torch.cat, F.interpolate, residual adds that are
                 not nn.Modules and therefore missing from layers.csv)
  roofline.png   roofline plot: one point per layer, area = share of time
  summary.md     peaks, ridge point, totals, memory-bound share, top-10 layers

Requirements: torch, matplotlib; torchvision + kfold_unet_variants.py (run from
~/YOLOv8) for --model unet; ultralytics for --model yolo.
"""
import argparse
import csv
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

SKIP_TYPES = (nn.Identity, nn.Dropout, nn.Dropout2d)
ACT_TYPES = (nn.ReLU, nn.ReLU6, nn.SiLU, nn.Sigmoid, nn.LeakyReLU,
             nn.Hardswish, nn.Hardsigmoid, nn.GELU, nn.Tanh, nn.Mish)
NORM_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)


# --------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------
def build_model(args):
    if args.model == "unet":
        # The project's own ResNetUNet (kfold_unet_variants.py): torchvision ResNet
        # encoder + U-Net decoder, 1-channel input, bg / mf / ian output.
        from kfold_unet_variants import CONFIGS, build_model as build_unet
        model = build_unet({**CONFIGS[args.unet_config], "pretrained": False})
        if args.weights:
            ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    else:
        if args.weights:
            from ultralytics import YOLO
            model = YOLO(args.weights).model
        else:
            from ultralytics.nn.tasks import SegmentationModel
            model = SegmentationModel(args.yolo_cfg, ch=args.in_ch, nc=args.classes,
                                      verbose=False)
        if args.fuse:
            model = model.fuse()  # folds BatchNorm into the preceding conv
    return model.float().eval()


# --------------------------------------------------------------------------
# Per-layer cost model
# --------------------------------------------------------------------------
def tensors_in(obj):
    """Collect all tensors from nested tuples / lists / dicts."""
    if torch.is_tensor(obj):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return [t for o in obj for t in tensors_in(o)]
    if isinstance(obj, dict):
        return [t for o in obj.values() for t in tensors_in(o)]
    return []


def op_type(mod):
    if isinstance(mod, nn.Conv2d):
        k = "x".join(str(v) for v in mod.kernel_size)
        if mod.groups > 1 and mod.groups == mod.in_channels:
            return f"Conv{k} depthwise"
        return f"Conv{k}"
    if isinstance(mod, nn.ConvTranspose2d):
        return "ConvT" + "x".join(str(v) for v in mod.kernel_size)
    return type(mod).__name__


def layer_flops(mod, ins, outs):
    """FLOPs for one call; one multiply-accumulate counts as 2 FLOPs."""
    out_n = sum(t.numel() for t in outs)
    x = ins[0] if ins else None
    if isinstance(mod, nn.Conv2d):
        kh, kw = mod.kernel_size
        return 2 * out_n * (mod.in_channels // mod.groups) * kh * kw
    if isinstance(mod, nn.ConvTranspose2d):
        kh, kw = mod.kernel_size
        return 2 * x.numel() * (mod.out_channels // mod.groups) * kh * kw
    if isinstance(mod, nn.Linear):
        return 2 * out_n * mod.in_features
    if isinstance(mod, NORM_TYPES):
        return 2 * out_n  # scale + shift at inference
    if isinstance(mod, (nn.MaxPool2d, nn.AvgPool2d)):
        k = mod.kernel_size
        k = k * k if isinstance(k, int) else k[0] * k[1]
        return out_n * k
    if isinstance(mod, (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
        return x.numel() if x is not None else 0
    if isinstance(mod, ACT_TYPES):
        return out_n
    return 0  # Upsample, Concat and other pure data movement


def layer_bytes(mod, ins, outs):
    """Bytes moved assuming each input, output and weight crosses DRAM once."""
    act = sum(t.numel() * t.element_size() for t in ins + outs)
    w = sum(p.numel() * p.element_size() for p in mod.parameters(recurse=False))
    w += sum(b.numel() * b.element_size() for b in mod.buffers(recurse=False))
    return act + w


# --------------------------------------------------------------------------
# Hooks
# --------------------------------------------------------------------------
class LayerProfiler:
    def __init__(self, model, device):
        self.cuda = device.type == "cuda"
        self.static = []          # per call: name, type, shapes, flops, bytes
        self.times = defaultdict(list)  # call index -> list of ms
        self._pending = []
        self._start = {}
        self._analyze = True
        self.handles = []
        for name, mod in model.named_modules():
            if len(list(mod.children())) == 0 and not isinstance(mod, SKIP_TYPES):
                self.handles.append(mod.register_forward_pre_hook(self._pre(name)))
                self.handles.append(mod.register_forward_hook(self._post(name)))

    def _now(self):
        if self.cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            return ev
        return time.perf_counter()

    def _pre(self, name):
        def hook(mod, inputs):
            self._start[name] = self._now()
        return hook

    def _post(self, name):
        def hook(mod, inputs, output):
            end = self._now()
            self._pending.append((self._start.pop(name), end))
            if self._analyze:
                ins, outs = tensors_in(inputs), tensors_in(output)
                params = sum(p.numel() for p in mod.parameters(recurse=False))
                self.static.append({
                    "name": name or "(root)",
                    "type": op_type(mod),
                    "in_shape": " ".join("x".join(map(str, t.shape)) for t in ins),
                    "out_shape": " ".join("x".join(map(str, t.shape)) for t in outs),
                    "params": params,
                    "flops": layer_flops(mod, ins, outs),
                    "bytes": layer_bytes(mod, ins, outs),
                })
        return hook

    def run(self, model, x, record):
        self._pending = []
        model(x)
        if self.cuda:
            torch.cuda.synchronize()
        if self._analyze:
            self._analyze = False
            self.n_calls = len(self.static)
        if not record:
            return
        if len(self._pending) != self.n_calls:
            raise RuntimeError("number of layer calls changed between passes")
        for i, (s, e) in enumerate(self._pending):
            ms = s.elapsed_time(e) if self.cuda else (e - s) * 1e3
            self.times[i].append(ms)

    def remove(self):
        for h in self.handles:
            h.remove()


# --------------------------------------------------------------------------
# Platform peaks
# --------------------------------------------------------------------------
def timed(fn, reps, cuda):
    if cuda:
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / 1e3
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return time.perf_counter() - t0


def measure_peaks(device, dtype):
    cuda = device.type == "cuda"
    # Compute peak: large square matmul.
    for n in ((8192, 4096, 2048) if cuda else (2048, 1024)):
        try:
            a = torch.randn(n, n, device=device, dtype=dtype)
            b = torch.randn(n, n, device=device, dtype=dtype)
            break
        except RuntimeError:  # out of memory
            if cuda:
                torch.cuda.empty_cache()
    reps = 10 if cuda else 3
    timed(lambda: a @ b, 2, cuda)
    peak_flops = 2 * n ** 3 * reps / timed(lambda: a @ b, reps, cuda)
    del a, b

    # Bandwidth: device-to-device copy of a large buffer (read + write).
    nbytes = (512 if cuda else 256) * 2 ** 20
    src = torch.empty(nbytes // 4, device=device, dtype=torch.float32)
    dst = torch.empty_like(src)
    reps = 20 if cuda else 5
    timed(lambda: dst.copy_(src), 2, cuda)
    peak_bw = 2 * nbytes * reps / timed(lambda: dst.copy_(src), reps, cuda)
    del src, dst
    if cuda:
        torch.cuda.empty_cache()
    return peak_flops, peak_bw


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["unet", "yolo"], required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--imgsz", type=int, default=None, help="default 512 (unet) / 1280 (yolo)")
    ap.add_argument("--height", type=int, default=None,
                    help="input height if not square (YOLO letterbox: 544 for 1024-wide panoramas, 896 for 1280-wide ROIs); default imgsz")
    ap.add_argument("--in-ch", type=int, default=None, help="default 1 (unet) / 3 (yolo)")
    ap.add_argument("--classes", type=int, default=None, help="default 3 (unet: bg, mf, ian) / 2 (yolo)")
    ap.add_argument("--yolo-cfg", default="yolov8n-seg.yaml")
    ap.add_argument("--unet-config", default="r34_imagenet_aug", help="key of kfold_unet_variants.CONFIGS")
    ap.add_argument("--fuse", action="store_true", help="yolo only: fold BN into conv")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--tf32", action="store_true", help="allow TF32 for fp32 matmul/conv")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--out", default="roofline_out")
    args = ap.parse_args()

    is_unet = args.model == "unet"
    args.imgsz = args.imgsz or (512 if is_unet else 1280)
    args.height = args.height or args.imgsz
    args.in_ch = args.in_ch or (1 if is_unet else 3)
    args.classes = args.classes or (3 if is_unet else 2)
    device = torch.device(args.device)
    cuda = device.type == "cuda"
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
        torch.backends.cudnn.benchmark = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("measuring platform peaks ...")
    peak_flops, peak_bw = measure_peaks(device, dtype)
    ridge = peak_flops / peak_bw
    print(f"  peak {peak_flops / 1e9:.0f} GFLOP/s, {peak_bw / 1e9:.0f} GB/s, ridge {ridge:.1f} FLOP/B")

    model = build_model(args).to(device=device, dtype=dtype)
    x = torch.randn(args.batch, args.in_ch, args.height, args.imgsz, device=device, dtype=dtype)

    with torch.inference_mode():
        # 1. End-to-end latency, no hooks.
        for _ in range(args.warmup):
            model(x)
        e2e = []
        for _ in range(args.runs):
            e2e.append(timed(lambda: model(x), 1, cuda) * 1e3)
        e2e_ms = statistics.median(e2e)
        print(f"end-to-end median latency: {e2e_ms:.2f} ms")

        # 2. Low-level op table (includes cat / interpolate / add).
        from torch.profiler import ProfilerActivity, profile
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
        with profile(activities=acts) as prof:
            for _ in range(10):
                model(x)
            if cuda:
                torch.cuda.synchronize()
        table = None
        for key in (["self_device_time_total", "self_cuda_time_total"] if cuda
                    else ["self_cpu_time_total"]):
            try:
                table = prof.key_averages().table(sort_by=key, row_limit=30)
                break
            except (KeyError, AttributeError, ValueError):
                continue
        (out / "aten_ops.txt").write_text(table or "profiler table unavailable\n")

        # 3. Per-layer timing with hooks.
        lp = LayerProfiler(model, device)
        lp.run(model, x, record=False)  # analysis pass
        for _ in range(args.warmup):
            lp.run(model, x, record=False)
        for _ in range(args.runs):
            lp.run(model, x, record=True)
        lp.remove()

    # ---- assemble rows ----
    rows = []
    for i, st in enumerate(lp.static):
        ms = statistics.median(lp.times[i])
        ai = st["flops"] / st["bytes"] if st["bytes"] else 0.0
        rows.append({
            "seq": i, **st,
            "time_ms": ms,
            "intensity_flop_per_byte": ai,
            "achieved_gflops": st["flops"] / (ms * 1e-3) / 1e9 if ms > 0 else 0.0,
            "bound": "memory" if ai < ridge else "compute",
        })
    layer_ms = sum(r["time_ms"] for r in rows)
    for r in rows:
        r["time_share"] = r["time_ms"] / layer_ms if layer_ms else 0.0

    with open(out / "layers.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    by_type = defaultdict(lambda: {"calls": 0, "time_ms": 0.0, "flops": 0, "bytes": 0})
    for r in rows:
        g = by_type[r["type"]]
        g["calls"] += 1
        g["time_ms"] += r["time_ms"]
        g["flops"] += r["flops"]
        g["bytes"] += r["bytes"]
    type_rows = []
    for t, g in sorted(by_type.items(), key=lambda kv: -kv[1]["time_ms"]):
        ai = g["flops"] / g["bytes"] if g["bytes"] else 0.0
        type_rows.append({"type": t, **g, "time_share": g["time_ms"] / layer_ms,
                          "intensity_flop_per_byte": ai,
                          "bound": "memory" if ai < ridge else "compute"})
    with open(out / "by_type.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(type_rows[0].keys()))
        w.writeheader()
        w.writerows(type_rows)

    # ---- roofline plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = [r for r in rows if r["flops"] > 0 and r["time_ms"] > 0]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    lo = min(r["intensity_flop_per_byte"] for r in pts) / 2
    hi = max(max(r["intensity_flop_per_byte"] for r in pts) * 2, ridge * 4)
    xs = [lo * (hi / lo) ** (k / 200) for k in range(201)]
    ax.plot(xs, [min(peak_flops, peak_bw * v) / 1e9 for v in xs], color="black", lw=1.5,
            label=f"roof: {peak_flops / 1e9:.0f} GFLOP/s, {peak_bw / 1e9:.0f} GB/s")
    ax.axvline(ridge, color="gray", ls="--", lw=1)
    types = [t["type"] for t in type_rows if any(r["type"] == t["type"] for r in pts)]
    cmap = plt.get_cmap("tab10")
    for k, t in enumerate(types):
        sel = [r for r in pts if r["type"] == t]
        ax.scatter([r["intensity_flop_per_byte"] for r in sel],
                   [r["achieved_gflops"] for r in sel],
                   s=[15 + 900 * r["time_share"] for r in sel],
                   color=cmap(k % 10), alpha=0.7, edgecolors="white", linewidths=0.5,
                   label=t)
    for r in sorted(pts, key=lambda r: -r["time_ms"])[:5]:
        ax.annotate(r["name"], (r["intensity_flop_per_byte"], r["achieved_gflops"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOP / byte)")
    ax.set_ylabel("Achieved performance (GFLOP/s)")
    ax.set_title(f"{args.model} {args.imgsz}x{args.height} (WxH), batch {args.batch}, "
                 f"{args.dtype}, {device.type} (point area = share of time)", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "roofline.png", dpi=200)

    # ---- summary ----
    total_flops = sum(r["flops"] for r in rows)
    total_bytes = sum(r["bytes"] for r in rows)
    total_params = sum(r["params"] for r in rows)
    mem_ms = sum(r["time_ms"] for r in rows if r["bound"] == "memory")
    gpu_name = torch.cuda.get_device_name(device) if cuda else "CPU"
    lines = [
        f"# Roofline summary: {args.model}",
        "",
        f"- Platform: {gpu_name}, {args.dtype}{', TF32 on' if args.tf32 else ''}",
        f"- Measured peak compute: {peak_flops / 1e9:.0f} GFLOP/s",
        f"- Measured memory bandwidth: {peak_bw / 1e9:.0f} GB/s",
        f"- Ridge point: {ridge:.1f} FLOP/byte (layers below it are memory-bound)",
        f"- Input: {args.batch}x{args.in_ch}x{args.height}x{args.imgsz} (NCHW), "
        f"params {total_params / 1e6:.2f} M",
        f"- Total compute: {total_flops / 1e9:.2f} GFLOPs per forward pass",
        f"- Total modelled traffic: {total_bytes / 1e6:.1f} MB; "
        f"whole-model intensity {total_flops / total_bytes:.1f} FLOP/byte",
        f"- End-to-end latency (no hooks, median of {args.runs}): {e2e_ms:.2f} ms",
        f"- Sum of per-layer times: {layer_ms:.2f} ms "
        f"(the gap to end-to-end is functional ops such as cat / interpolate / add, "
        f"see aten_ops.txt, plus hook overhead)",
        f"- **Share of layer time spent in memory-bound layers: {mem_ms / layer_ms:.1%}**",
        "",
        "## Time by operator type",
        "",
        "| type | calls | time ms | share | GFLOPs | intensity FLOP/B | bound |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for t in type_rows:
        lines.append(f"| {t['type']} | {t['calls']} | {t['time_ms']:.3f} | {t['time_share']:.1%} | "
                     f"{t['flops'] / 1e9:.3f} | {t['intensity_flop_per_byte']:.1f} | {t['bound']} |")
    lines += [
        "",
        "## Top 10 layers by time",
        "",
        "| # | layer | type | out shape | time ms | share | intensity FLOP/B | achieved GFLOP/s | bound |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in sorted(rows, key=lambda r: -r["time_ms"])[:10]:
        lines.append(f"| {r['seq']} | {r['name']} | {r['type']} | {r['out_shape']} | "
                     f"{r['time_ms']:.3f} | {r['time_share']:.1%} | "
                     f"{r['intensity_flop_per_byte']:.1f} | {r['achieved_gflops']:.0f} | {r['bound']} |")
    lines += [
        "",
        "## Caveats",
        "",
        "- Bytes assume every input, output and weight crosses DRAM once per layer; "
        "on-chip cache reuse between layers is ignored, so small layers' true "
        "intensity can be higher than shown.",
        "- At batch 1 the time of tiny layers includes kernel-launch gaps, which is "
        "real eager-mode latency but not a property of the hardware roof.",
        "- Unfused BatchNorm and activations appear as separate memory-bound layers; "
        "a deployment compiler would fuse them into the convolution (try --fuse for YOLO).",
        "- The roof is this server's measured peak, not an edge device.",
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"done: results in {out.resolve()}")


if __name__ == "__main__":
    main()
