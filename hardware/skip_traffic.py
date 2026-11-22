"""U-Net skip-connection and residual traffic per memory level (2026-11-22, analysis only, CPU, random weights).

From the RISC-V project's design notes ("Ideas taken from Merrimac and Imagine", item 4): a skip feature map is
produced by the encoder and read much later by the decoder (long-range producer-consumer locality, Merrimac's SRF
level), and a ResNet block's input must stay until its residual add. accel_explore.py adds all skips at once
(2 x total if SRAM < total skips + max working set / 2); this script follows every skip and residual through the
layer order instead.

Model (INT8, 1 byte per element; no image or patient data, random weights, shapes only):
  - Layer order and shapes come from forward hooks on the project's ResNetUNet (kfold_unet_variants.py).
  - Per-layer DRAM traffic = accel_model.dram_traffic(layer, SRAM left over), the same tiling model as
    accel_explore.py, minus 2 x act_out when the next layer reads it directly and both fit (accel_explore's rule).
  - A skip (or a residual identity) kept on chip occupies its bytes for its whole lifetime, so the layers in
    between have less SRAM. A spilled one costs 2 x bytes (written at production, read at use; an upper bound for
    residuals, whose input may already be in DRAM).
  - Which to keep: greedy, smallest first; one is kept only if it lowers the total traffic.
  - Peak live bytes = max over layers of (input + output + weights + every skip / identity alive): the SRAM needed
    to run the whole network without any DRAM traffic beyond the input image and the weights.

usage: python -m hardware.skip_traffic [--tiling v2]   (writes compare_results/roofline/skip_traffic/{plans.csv,summary.md})
--tiling v2 (2026-11-22): per-layer traffic from accel_lowbit.dram_traffic_lb2, the tiling fixed after the Timeloop
cross-check (EXPERIMENTS.md); outputs in skip_traffic/v2/, the default (v1) outputs are unchanged.
"""
import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock

from hardware.accel_model import dram_traffic
from stage2.kfold_unet_variants import CONFIGS, build_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "compare_results" / "roofline" / "skip_traffic"
MODELS = {"r34_imagenet_aug": "main model (head.0.block.3 = the RISC-V benchmark layer)",
          "r34_imagenet_aug_slimhalf": "slim decoder (deployed since 2026-11-18)"}
SRAM_KB = [32, 128, 256, 512, 1024, 2048, 4096, 8192]
TRAFFIC = dram_traffic


def trace(config):
    model = build_model({**CONFIGS[config], "pretrained": False}).eval()
    layers, lives, hooks = [], [], []

    def conv_hook(name):
        def hook(mod, inp, out):
            i, o = inp[0], out
            m, n = o.shape[2] * o.shape[3], o.shape[1]
            mac = m * n * (mod.in_channels // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1]
            layers.append({"name": name, "M": m, "N": n, "K": mac // (m * n), "Cin": i.shape[1], "mac": mac,
                           "in_hw": tuple(i.shape[2:]), "out_hw": tuple(o.shape[2:]),
                           "weights": mod.weight.numel() + (mod.bias.numel() if mod.bias is not None else 0), "act_in": i.numel(), "act_out": o.numel()})
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(conv_hook(name)))

    # skips: produced when stem / layers[k] finishes, used by the first conv of the decoder that pops them
    produced = {}

    def produce(key):
        def hook(mod, inp, out):
            produced[key] = (len(layers), out.numel())   # alive from the next layer on
        return hook

    hooks.append(model.stem.register_forward_hook(produce("stem")))
    for k, layer in enumerate(model.layers):
        hooks.append(layer.register_forward_hook(produce(f"layers.{k}")))
    order = [f"layers.{k}" for k in range(len(model.layers) - 1)][::-1] + ["stem"]

    def consume(key):
        def hook(mod, inp):
            start, size = produced[key]
            lives.append({"kind": "skip", "name": key, "bytes": size, "start": start, "end": len(layers)})
        return hook

    for dec, key in zip(model.decoders, order):
        hooks.append(dec.register_forward_pre_hook(consume(key)))

    # residual identity of each BasicBlock: alive from after its first conv until conv2 (the add)
    for name, mod in model.named_modules():
        if isinstance(mod, BasicBlock):
            def pre(mod, inp, name=name):
                mod._start = len(layers)
            def post(mod, inp, out, name=name):
                lives.append({"kind": "residual", "name": name, "bytes": inp[0].numel(),
                              "start": mod._start + 1, "end": len(layers)})
            hooks += [mod.register_forward_pre_hook(pre), mod.register_forward_hook(post)]

    with torch.inference_mode():
        model(torch.zeros(1, 1, *CONFIGS[config]["size"]))
    for h in hooks:
        h.remove()
    # a conv whose input is the identity itself (the block's downsample) does not need it as extra storage
    for lv in lives:
        lv["layers"] = [j for j in range(lv["start"], lv["end"])
                        if not (lv["kind"] == "residual" and layers[j]["name"].endswith("downsample.0"))]
    return layers, lives


def total_traffic(layers, lives, kept, sram):
    held = [0] * len(layers)
    for lv in lives:
        if lv["name"] in kept:
            for j in lv["layers"]:
                held[j] += lv["bytes"]
    layer_t = 0.0
    for i, l in enumerate(layers):
        avail = sram - held[i]
        if avail <= 0:
            return math.inf, None
        t = TRAFFIC(l, avail)
        if t == math.inf:
            return math.inf, None
        nxt = layers[i + 1] if i + 1 < len(layers) else None
        if (nxt and nxt["in_hw"] == l["out_hw"] and l["act_out"] <= nxt["act_in"]
                and l["act_out"] + nxt["weights"] + nxt["act_out"] <= sram - max(held[i], held[i + 1])):
            t -= 2 * l["act_out"]
        layer_t += t
    skip_t = sum(2 * lv["bytes"] for lv in lives if lv["kind"] == "skip" and lv["name"] not in kept)
    res_t = sum(2 * lv["bytes"] for lv in lives if lv["kind"] == "residual" and lv["name"] not in kept)
    return layer_t + skip_t + res_t, (layer_t, skip_t, res_t)


def plan(layers, lives, sram):
    kept = set()
    best, parts = total_traffic(layers, lives, kept, sram)
    for lv in sorted(lives, key=lambda lv: lv["bytes"]):
        t, p = total_traffic(layers, lives, kept | {lv["name"]}, sram)
        if t < best:
            kept.add(lv["name"])
            best, parts = t, p
    return best, parts, kept


def main():
    global OUT, TRAFFIC
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiling", choices=["v1", "v2"], default="v1")
    args = ap.parse_args()
    if args.tiling == "v2":
        from hardware.accel_lowbit import dram_traffic_lb2
        OUT, TRAFFIC = OUT / "v2", (lambda l, sram: dram_traffic_lb2(l, sram, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [f"# U-Net skip / residual traffic per SRAM size (INT8, analysis only, tiling {args.tiling})", "",
             "Generated by `skip_traffic.py` (see its docstring for the model). MB = 10^6 bytes; one 512 x 512 ROI.", ""]
    rows = []
    for config, desc in MODELS.items():
        layers, lives = trace(config)
        skips = [lv for lv in lives if lv["kind"] == "skip"]
        res = [lv for lv in lives if lv["kind"] == "residual"]
        live = [l["act_in"] + l["act_out"] + l["weights"]
                + sum(lv["bytes"] for lv in lives if j in lv["layers"]) for j, l in enumerate(layers)]
        peak = max(range(len(layers)), key=lambda j: live[j])
        gmac = sum(l["mac"] for l in layers) / 1e9
        wmb = sum(l["weights"] for l in layers) / 1e6
        lines += [f"## {config}: {desc}", "",
                  f"- {len(layers)} convolutions, {gmac:.2f} G MACs, INT8 weights {wmb:.2f} MB.",
                  f"- Peak live bytes {live[peak] / 1e6:.2f} MB at `{layers[peak]['name']}` (SRAM needed to keep everything on chip).",
                  "", "| skip | bytes | produced after | used by | layers in between |", "| --- | ---: | --- | --- | ---: |"]
        for lv in skips:
            lines.append(f"| `{lv['name']}` | {lv['bytes'] / 1e6:.3f} MB | `{layers[lv['start'] - 1]['name']}` | "
                         f"`{layers[lv['end']]['name']}` | {len(lv['layers'])} |")
        lines += ["", f"Skips total {sum(s['bytes'] for s in skips) / 1e6:.2f} MB; {len(res)} residual identities, "
                  f"largest {max(r['bytes'] for r in res) / 1e6:.2f} MB.", "",
                  "| SRAM | total DRAM MB | layers | skips | residuals | skip share | skips kept on chip |",
                  "| ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
        for kb in SRAM_KB:
            t, parts, kept = plan(layers, lives, kb * 1024)
            if parts is None:
                lines.append(f"| {kb} KB | no tile fits | | | | | |")
                continue
            lt, st, rt = parts
            ks = ", ".join(f"`{n}`" for n in [s["name"] for s in skips] if n in kept) or "none"
            nres = sum(1 for r in res if r["name"] in kept)
            lines.append(f"| {kb} KB | {t / 1e6:.1f} | {lt / 1e6:.1f} | {st / 1e6:.1f} | {rt / 1e6:.1f} | "
                         f"{st / t:.0%} | {ks}; residuals {nres} / {len(res)} |")
            rows.append({"config": config, "sram_kb": kb, "total_mb": round(t / 1e6, 3), "layers_mb": round(lt / 1e6, 3),
                         "skips_mb": round(st / 1e6, 3), "residuals_mb": round(rt / 1e6, 3),
                         "skips_kept": " ".join(n for n in [s["name"] for s in skips] if n in kept), "residuals_kept": nres})
        lines += ["", "| layer | in | out | MACs | weights | live bytes |", "| --- | --- | --- | ---: | ---: | ---: |"]
        for j, l in enumerate(layers):
            lines.append(f"| `{l['name']}` | {l['Cin']}x{l['in_hw'][0]}x{l['in_hw'][1]} | {l['N']}x{l['out_hw'][0]}x{l['out_hw'][1]} | "
                         f"{l['mac'] / 1e6:,.0f} M | {l['weights']:,} | {live[j] / 1e6:.2f} MB |")
        lines.append("")
    with open(OUT / "plans.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
