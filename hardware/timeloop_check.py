"""Timeloop cross-check of the L2.5 analysis model (2026-11-22; criteria in EXPERIMENTS.md, set before the run).

Timeloop v3.0.3 (~/tools/timeloop, env ~/conda-envs/timeloop), built-in energy model, legacy "arch" format:
DRAM -> GlobalBuffer 1 MB (8-bit words; weights, inputs, outputs) -> 64 x 64 PEs with one weight register each
(weight-stationary). Spatial factors only on C (rows) and K (columns). Two mapper runs per layer:
  ws   : timeloop-model with a fixed WS mapping (only when C is a multiple of 64): GLB spatial C64 x K64,
         GLB temporal R S, Q in the PE register (weight held across a row), P C/64 K/64 at DRAM
         -> C1 cycles against L2.5's ceil(C*9/64) * ceil(K/64) * P*Q
         (the mapper with spatial forced to C64 K64 ran 90+ min on L1 without finishing, so the mapping is given)
  free : spatial C and K free (each <= 64), at most SEARCH valid mappings per thread, objective last-level accesses -> C2 DRAM bytes against accel_model.dram_traffic
usage: python -m hardware.timeloop_check [--search N] [--layers name ...] [--summary FILE]
       (work dirs and summary in compare_results/roofline/timeloop_check/; the 2026-11-22 results used
        --search 3000 for all four layers, then 30000 for layers.3.1.conv1 and 300000 for decoders.0.block.0)
The summary also lists the compulsory traffic (weights, padded input and output each moved once).
"""
import argparse
import math
import re
import subprocess
from pathlib import Path

from hardware.accel_model import dram_traffic, load_layers

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "compare_results" / "roofline" / "timeloop_check"
TL = Path.home() / "tools" / "timeloop" / "bin"
ENV = Path.home() / "conda-envs" / "timeloop"
LAYERS = ["layers.0.0.conv1", "layers.3.1.conv1", "decoders.3.block.0", "decoders.0.block.0"]
SRAM = 1024 * 1024
ARRAY = 64
SEARCH = 3000   # valid mappings per mapper thread (24 threads); unlimited (0) ran 99+ min on L2 without finishing


def yaml_for(l, mode):
    c, k = l["Cin"], l["N"]
    p, q = l["out_hw"]
    r = s = int(round(math.sqrt(l["K"] // c)))
    spatial = "factors: P1 Q1 R1 S1 N1"
    metrics = "  - last-level-accesses\n  - delay"
    tail = ""
    if mode == "ws":
        keep = "    keep: [Weights, Inputs, Outputs]\n    bypass: []"
        tail = f"""mapping:
  - target: Registers
    type: datatype
    keep: [Weights]
    bypass: [Inputs, Outputs]
  - target: GlobalBuffer
    type: datatype
{keep}
  - target: DRAM
    type: datatype
{keep}
  - target: Registers
    type: temporal
    factors: R1 S1 P1 Q{q} C1 K1 N1
    permutation: QRSPCKN
  - target: GlobalBuffer
    type: spatial
    factors: R1 S1 P1 Q1 C{ARRAY} K{ARRAY} N1
    permutation: CKRSPQN
    split: 1
  - target: GlobalBuffer
    type: temporal
    factors: R{r} S{s} P1 Q1 C1 K1 N1
    permutation: RSCKPQN
  - target: DRAM
    type: temporal
    factors: R1 S1 P{p} Q1 C{c // ARRAY} K{k // ARRAY} N1
    permutation: PCKRSQN
"""
    return f"""mapper:
  algorithm: random-pruned
  optimization-metrics:
{metrics}
  num-threads: 24
  timeout: 3000
  victory-condition: 800
  search-size: {SEARCH}
  log-stats: false
  log-suboptimal: false
arch:
  arithmetic:
    instances: {ARRAY * ARRAY}
    meshX: {ARRAY}
    word-bits: 8
  storage:
  - name: Registers
    entries: 1
    instances: {ARRAY * ARRAY}
    meshX: {ARRAY}
    word-bits: 8
  - name: GlobalBuffer
    sizeKB: {SRAM // 1024}
    instances: 1
    word-bits: 8
    block-size: 8
  - name: DRAM
    technology: DRAM
    instances: 1
    word-bits: 8
    block-size: 8
    bandwidth: 1000.0
mapspace:
  constraints:
  - target: Registers
    type: datatype
    keep: [Weights]
    bypass: [Inputs, Outputs]
  - target: Registers
    type: temporal
    factors: R1 S1 C1 K1 N1
  - target: GlobalBuffer
    type: spatial
    {spatial}
    permutation: CKPQRSN
problem:
  shape: cnn-layer
  R: {r}
  S: {s}
  P: {p}
  Q: {q}
  C: {c}
  K: {k}
  N: 1
  Wstride: 1
  Hstride: 1
""" + tail


def run(l, mode):
    d = OUT / f"{l['name']}_{mode}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "in.yaml").write_text(yaml_for(l, mode))
    env = {"PATH": f"{ENV / 'bin'}:/usr/bin:/bin", "LD_LIBRARY_PATH": f"{ENV / 'lib'}:{TL.parent / 'lib'}", "HOME": str(Path.home())}
    tool = "timeloop-model" if mode == "ws" else "timeloop-mapper"
    subprocess.run([str(TL / tool), "in.yaml"], cwd=d, env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=7200)
    return (d / f"{tool}.stats.txt").read_text()


def parse(stats):
    cycles = int(re.search(r"^Cycles\s*:\s*(\d+)", stats, re.M).group(1))
    def level(name):
        block = stats.split(f"=== {name} ===")[1].split("=== ")[0]
        per = {}
        for ds in ("Weights", "Inputs", "Outputs"):
            m = re.search(rf"{ds}:\s*\n(.*?)(?:\n    [A-Z][a-z]+:\s*\n|\Z)", block, re.S)
            body = m.group(1) if m else ""
            get = lambda key: sum(int(float(v)) for v in re.findall(rf"{key}\s*\(per-instance\)\s*:\s*([\d.]+)", body)) if body else 0
            per[ds] = {"reads": get("Scalar reads"), "fills": get("Scalar fills"), "updates": get("Scalar updates")}
        return per
    return cycles, level("DRAM"), level("GlobalBuffer")


def compulsory(l):
    c, k = l["Cin"], l["N"]
    p, q = l["out_hw"]
    r = int(round(math.sqrt(l["K"] // c)))
    return c * k * r * r + c * (p + r - 1) * (q + r - 1) + k * p * q


def main():
    global SEARCH
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", type=int, default=SEARCH)
    ap.add_argument("--layers", nargs="+", default=LAYERS)
    ap.add_argument("--summary", default="summary.md")
    args = ap.parse_args()
    SEARCH = args.search
    OUT.mkdir(parents=True, exist_ok=True)
    layers = {l["name"]: l for l in load_layers("stage2_unet_r34_slimhalf")}
    rows = []
    for name in args.layers:
        l = layers[name]
        res = {"name": name, "C": l["Cin"], "K": l["N"], "PQ": l["M"], "compulsory": compulsory(l)}
        if l["Cin"] % ARRAY == 0 and l["N"] % ARRAY == 0:
            cyc, _, _ = parse(run(l, "ws"))
            res["tl_cycles"] = cyc
        res["l25_cycles"] = math.ceil(l["K"] / ARRAY) * math.ceil(l["N"] / ARRAY) * l["M"]
        cyc_free, dram, glb = parse(run(l, "free"))
        res["tl_cycles_free"] = cyc_free
        res["tl_dram"] = sum(v["reads"] + v["updates"] for v in dram.values())
        res["tl_dram_split"] = {k: v["reads"] + v["updates"] for k, v in dram.items()}
        res["tl_glb"] = sum(v["reads"] + v["fills"] + v["updates"] for v in glb.values())
        res["l25_dram"] = dram_traffic(l, SRAM)
        rows.append(res)
        print(res, flush=True)
    lines = [f"# Timeloop cross-check of L2.5 (slim model layers, INT8, 64 x 64 WS, 1 MB, search {SEARCH})", "",
             "Generated by `timeloop_check.py`; criteria in EXPERIMENTS.md (set before the run).", "",
             "| layer | C1 cycles TL / L2.5 | C1 | C2 DRAM bytes TL | L2.5 | L2.5 / TL | C2 | compulsory | TL DRAM W / I / O | TL cycles (traffic mapping) |",
             "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- | ---: |"]
    for r in rows:
        if "tl_cycles" in r:
            ratio = r["tl_cycles"] / r["l25_cycles"]
            c1 = f"{r['tl_cycles']:,} / {r['l25_cycles']:,} ({ratio:.3f})", "pass" if abs(ratio - 1) <= 0.05 else "FAIL"
        else:
            c1 = f"{r['tl_cycles_free']:,} / {r['l25_cycles']:,} (report only)", "-"
        q = r["l25_dram"] / r["tl_dram"]
        c2 = "pass" if 0.9 <= q <= 1.25 else "FAIL"
        sp = r["tl_dram_split"]
        lines.append(f"| `{r['name']}` | {c1[0]} | {c1[1]} | {r['tl_dram']:,} | {r['l25_dram']:,} | {q:.2f} | {c2} | "
                     f"{r['compulsory']:,} | {sp['Weights']:,} / {sp['Inputs']:,} / {sp['Outputs']:,} | {r['tl_cycles_free']:,} |")
    (OUT / args.summary).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
