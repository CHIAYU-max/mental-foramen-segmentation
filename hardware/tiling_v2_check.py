"""Check of the fixed L2.5 tiling (accel_lowbit.dram_traffic_lb2) against Timeloop (2026-11-22; criteria V1 / V2 in
EXPERIMENTS.md "Timeloop 核對之後的三項", set before the run).
  V1: the four Timeloop layers, INT8, 1 MB: v2 within -10% .. +25% of Timeloop's converged minimum DRAM traffic.
  V2: every layer of the main and the slim model, every SRAM size and bit-width / sparsity configuration:
      v2 <= old tiling and v2 >= weights + act_in + act_out (scaled); stops on the first violation.
usage: python -m hardware.tiling_v2_check
"""
import math

from hardware.accel_lowbit import CONFIGS, dram_traffic_lb, dram_traffic_lb2, factors
from hardware.accel_model import load_layers

TIMELOOP = {"layers.0.0.conv1": 2_167_040, "layers.3.1.conv1": 2_656_256,
            "decoders.3.block.0": 9_055_232, "decoders.0.block.0": 1_903_616}
SRAM_KB = [64, 128, 256, 512, 1024, 2048, 4096, 8192]


def main():
    slim = {l["name"]: l for l in load_layers("stage2_unet_r34_slimhalf")}
    print("| layer | Timeloop | L2.5 old | old / TL | L2.5 v2 | v2 / TL | V1 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for name, tl in TIMELOOP.items():
        l = slim[name]
        old, new = dram_traffic_lb(l, 1024 * 1024, 1, 1), dram_traffic_lb2(l, 1024 * 1024, 1, 1)
        ok = 0.9 <= new / tl <= 1.25
        print(f"| `{name}` | {tl:,} | {old:,.0f} | {old / tl:.2f} | {new:,.0f} | {new / tl:.2f} | {'pass' if ok else 'FAIL'} |")
    n = 0
    for model in ("stage2_unet_r34", "stage2_unet_r34_slimhalf"):
        for l in load_layers(model):
            for kb in SRAM_KB:
                for label, wb, ab, sp in CONFIGS:
                    fw, fa = factors(wb, ab, sp)
                    old, new = dram_traffic_lb(l, kb * 1024, fw, fa), dram_traffic_lb2(l, kb * 1024, fw, fa)
                    floor = l["weights"] * fw + (l["act_in"] + l["act_out"]) * fa
                    assert new <= old * (1 + 1e-12), (model, l["name"], kb, label, old, new)
                    assert new >= floor * (1 - 1e-12), (model, l["name"], kb, label, floor, new)
                    n += 1
    print(f"V2: {n} cases, v2 <= old and v2 >= compulsory (no padding) in all")


if __name__ == "__main__":
    main()
