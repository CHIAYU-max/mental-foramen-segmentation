"""Timeloop follow-ups (2) and (3) after the L2.5 cross-check (2026-11-22; criteria S1, P1, P2 in EXPERIMENTS.md
"Timeloop 核對之後的三項", set before the run). Same architecture and layers as timeloop_check.py.

(2) S1  full-speed minimum traffic: timeloop-mapper with the GlobalBuffer spatial factors fixed to C64 x K64 (every
        mapping then runs at the full-array cycle count), objective last-level accesses, --search valid mappings
        per thread; v2 tiling (accel_lowbit.dram_traffic_lb2) must be within -10% .. +25% of Timeloop's best.
(3) P1 / P2  2:4 with Sparseloop: the fixed WS mapping of timeloop_check (GlobalBuffer spatial C64 x K64), but the
        PE register holds the weight and the current input (2 entries; Q moves from the register to the
        GlobalBuffer temporal loop, innermost) so that skipping at the register can remove the MAC of a zero
        weight. Dense and 2:4 run on that same architecture with timeloop-model:
          weights fixed-structured density 0.5; compressed (UOP ranks + CP with 2-bit coordinates) in DRAM and the
          GlobalBuffer (the number of UOP ranks is the smallest that Timeloop accepts, found per layer);
          skipping of Inputs on Weights at the GlobalBuffer and at the register.
        P1: Timeloop's weight DRAM bytes (values + metadata) within +-10% of L2.5's weights x 0.5 x 10 / 8.
        P2: Timeloop cycles 2:4 / dense within +-10% of L2.5's ceil(ceil(K/2)/64) / ceil(K/64).
usage: python -m hardware.timeloop_followup s1 [--search N] [--layers ...] | p
outputs: compare_results/roofline/timeloop_check/<layer>_{wsmap,ws2,ws24}/ and followup_{s1,p}.md
"""
import argparse
import math
import re
import subprocess

from hardware import timeloop_check as t
from hardware.accel_lowbit import dram_traffic_lb, dram_traffic_lb2
from hardware.accel_model import load_layers

LAYERS = ["layers.0.0.conv1", "layers.3.1.conv1", "decoders.0.block.0"]
ENV = {"PATH": f"{t.ENV / 'bin'}:/usr/bin:/bin", "LD_LIBRARY_PATH": f"{t.ENV / 'lib'}:{t.TL.parent / 'lib'}",
       "HOME": str(t.Path.home())}


def sub(s, a, b):
    assert a in s, a
    return s.replace(a, b)


def run_yaml(name, tag, text, tool):
    d = t.OUT / f"{name}_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "in.yaml").write_text(text)
    r = subprocess.run([str(t.TL / tool), "in.yaml"], cwd=d, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, timeout=7200)
    return r.returncode, r.stdout, d / f"{tool}.stats.txt"


def wsmap_yaml(l):
    return sub(t.yaml_for(l, "free"), "    factors: P1 Q1 R1 S1 N1\n    permutation: CKPQRSN",
               f"    factors: P1 Q1 R1 S1 C{t.ARRAY} K{t.ARRAY} N1\n    permutation: CKPQRSN")


def ws2_yaml(l):
    """Fixed WS mapping, register holds weight + current input, Q innermost at the GlobalBuffer."""
    s = t.yaml_for(l, "ws")
    q = l["out_hw"][1]
    s = sub(s, "  - name: Registers\n    entries: 1\n", "  - name: Registers\n    entries: 2\n    decompression_supported: true\n")
    s = s.replace("    keep: [Weights]\n    bypass: [Inputs, Outputs]", "    keep: [Weights, Inputs]\n    bypass: [Outputs]")
    s = sub(s, f"    factors: R1 S1 P1 Q{q} C1 K1 N1\n    permutation: QRSPCKN",
            "    factors: R1 S1 P1 Q1 C1 K1 N1\n    permutation: QRSPCKN")
    r = int(round(math.sqrt(l["K"] // l["Cin"])))
    s = sub(s, f"    factors: R{r} S{r} P1 Q1 C1 K1 N1\n    permutation: RSCKPQN",
            f"    factors: R{r} S{r} P1 Q{q} C1 K1 N1\n    permutation: QRSCKPN")
    return s


def ws24_yaml(l, nd, ng):
    s = ws2_yaml(l)
    md = "    metadata_storage_width: 8\n    metadata_datawidth: 2\n    metadata_block_size: 4\n"
    flags = "    decompression_supported: true\n    compression_supported: true\n"
    s = sub(s, "  - name: GlobalBuffer\n    sizeKB: 1024\n",
            "  - name: GlobalBuffer\n    sizeKB: 1024\n" + md + "    metadata_storage_depth: 1048576\n" + flags)
    s = sub(s, "    technology: DRAM\n", "    technology: DRAM\n" + md + flags)
    s = sub(s, "  Hstride: 1\n", "  Hstride: 1\n  densities:\n    Weights:\n      distribution: fixed-structured\n      density: 0.5\n")

    def fmt(n):
        ranks = "".join("        - format: UOP\n" for _ in range(n - 1))
        return ("    representation-format:\n      data-spaces:\n      - name: Weights\n        ranks:\n"
                f"{ranks}        - format: CP\n          metadata-word-bits: 2\n")
    skip = "    action-optimization:\n    - type: skipping\n      options:\n      - target: Inputs\n        condition-on: [Weights]\n"
    return (s + "sparse_optimizations:\n  targets:\n  - name: DRAM\n" + fmt(nd) + "  - name: GlobalBuffer\n" + fmt(ng)
            + skip + "  - name: Registers\n" + skip)


def weight_dram_bytes(stats):
    """Weights at DRAM: actual scalar reads (8-bit values) + metadata and payload format reads (bits from the spec:
    CP coordinates 2 bits; UOP offsets at the metadata storage width, 8 bits)."""
    block = stats.split("=== DRAM ===")[1].split("=== ")[0]
    w = block.split("Weights:")[1].split("Inputs:")[0]
    data = int(re.search(r"Actual scalar reads \(per-instance\)\s*:\s*(\d+)", w).group(1))
    fmt = w.split("Actual scalar format reads (per-instance)")[1].split("Gated scalar format reads")[0]
    md = pl = 0
    for rank, m, p in re.findall(r"Rank (\d+) \(metadata, payload\): \((\d+),\s*(\d+)\)", fmt):
        md += int(m) * 2
        pl += int(p) * 8
    return data + (md + pl) / 8, data, (md + pl) / 8


def actual(stats, ds, kind):
    """DRAM actual scalar reads / updates of a dataspace in Sparseloop stats ("Actual scalar ..." lines; the dense
    parser in timeloop_check reads "Scalar ..." and gives 0 there)."""
    block = stats.split("=== DRAM ===")[1].split("=== ")[0]
    part = block.split(f"{ds}:")[1]
    return int(re.search(rf"Actual scalar {kind} \(per-instance\)\s*:\s*(\d+)", part).group(1))


def s1(args):
    t.SEARCH = args.search
    slim = {l["name"]: l for l in load_layers("stage2_unet_r34_slimhalf")}
    lines = [f"# Follow-up S1: full-speed minimum DRAM traffic (spatial fixed C64 x K64, search {args.search})", "",
             "| layer | TL DRAM | TL cycles | ideal cycles | L2.5 v2 | v2 / TL | S1 | L2.5 old | old / TL |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |"]
    for name in args.layers:
        l = slim[name]
        rc, out, st = run_yaml(name, "wsmap", wsmap_yaml(l), "timeloop-mapper")
        assert rc == 0, out[-500:]
        cyc, dram, _ = t.parse(st.read_text())
        tl = sum(v["reads"] + v["updates"] for v in dram.values())
        ideal = math.ceil(l["K"] / t.ARRAY) * math.ceil(l["N"] / t.ARRAY) * l["M"]
        new, old = dram_traffic_lb2(l, t.SRAM, 1, 1), dram_traffic_lb(l, t.SRAM, 1, 1)
        ok = "pass" if 0.9 <= new / tl <= 1.25 else "FAIL"
        lines.append(f"| `{name}` | {tl:,} | {cyc:,} | {ideal:,} | {new:,.0f} | {new / tl:.2f} | {ok} | {old:,.0f} | {old / tl:.2f} |")
        print(lines[-1], flush=True)
    (t.OUT / f"followup_s1_search{args.search}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def p(args):
    slim = {l["name"]: l for l in load_layers("stage2_unet_r34_slimhalf")}
    lines = ["# Follow-up P1 / P2: 2:4 with Sparseloop (fixed WS mapping, register holds weight + input)", "",
             "| layer | dense cycles | 2:4 cycles | TL ratio | L2.5 ratio | P2 | TL weight DRAM B (values + metadata) | L2.5 2:4 weight B | TL / L2.5 | P1 | UOP ranks DRAM / GLB | TL inputs / outputs DRAM B (2:4) |",
             "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- | --- |"]
    for name in args.layers:
        l = slim[name]
        rc, out, st = run_yaml(name, "ws2", ws2_yaml(l), "timeloop-model")
        assert rc == 0, out[-500:]
        dense_cyc, _, _ = t.parse(st.read_text())
        found = None
        for nd in range(1, 9):
            for ng in range(1, 9):
                rc, out, st = run_yaml(name, "ws24", ws24_yaml(l, nd, ng), "timeloop-model")
                if rc == 0:
                    found = (nd, ng)
                    break
            if found:
                break
        assert found, out[-500:]
        stats = st.read_text()
        cyc, dram, _ = t.parse(stats)
        wb, wdata, wmeta = weight_dram_bytes(stats)
        k = l["K"]
        l25_ratio = math.ceil(math.ceil(k / 2) / t.ARRAY) / math.ceil(k / t.ARRAY)
        tl_ratio = cyc / dense_cyc
        p2 = "pass" if abs(tl_ratio / l25_ratio - 1) <= 0.1 else "FAIL"
        l25_w = l["weights"] * 0.5 * 10 / 8
        p1 = "pass" if abs(wb / l25_w - 1) <= 0.1 else "FAIL"
        io = f"{actual(stats, 'Inputs', 'reads'):,} / {actual(stats, 'Outputs', 'updates'):,}"
        lines.append(f"| `{name}` | {dense_cyc:,} | {cyc:,} | {tl_ratio:.3f} | {l25_ratio:.3f} | {p2} | {wb:,.0f} ({wdata:,} + {wmeta:,.0f}) | "
                     f"{l25_w:,.0f} | {wb / l25_w:.3f} | {p1} | {found[0] - 1} / {found[1] - 1} | {io} |")
        print(lines[-1], flush=True)
    (t.OUT / "followup_p.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["s1", "p"])
    ap.add_argument("--search", type=int, default=30000)
    ap.add_argument("--layers", nargs="+", default=LAYERS)
    args = ap.parse_args()
    s1(args) if args.what == "s1" else p(args)


if __name__ == "__main__":
    main()
