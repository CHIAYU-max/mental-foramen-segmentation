"""2026-11-15: OpenVINO CPU 逐層剖析(PERF_COUNT)U-Net R34/512,合成影像(不需要病人影像)。

目的: (1) 確認 OpenVINO 編譯時真的有把 Conv / BN / ReLU / Add / FakeQuantize 融合;
      (2) 看 INT8(正式部署,種子 0 第 0 折)與 FP32 哪幾層最花時間。
只看同一次量測內的比例,不跨時段比較絕對毫秒數(筆電排程差約 ±25%)。

用法(從 repo 根目錄): python -m deployment.unet_perf_count
輸出: compare_results/perf_count/<主機>/(summary.md、nodes_<model>.csv)
"""
import csv
import platform
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import openvino as ov

import deployment.no_throttle as no_throttle

no_throttle.disable()  # 2026-11-17: 和其他計時程式一致(關背景節流、記錄計時環境到 timing_env_log.csv)

from stage2.compare_yolo_unet import ROOT

MODELS = {
    "int8": ROOT / "compare_results/int8_pano/models/s0_fold0_int8.xml",  # 正式部署用的 U-Net(種子 0 第 0 折)
    "fp32": ROOT / "compare_results/int8" / platform.node() / "unet_fp32.xml",  # 10/26 匯出的 FP32(第 0 折)
}
SHAPE = (1, 1, 512, 512)
WARMUP, RUNS, ROUNDS = 20, 100, 2
OUT = ROOT / "compare_results/perf_count" / platform.node()


def compile_model(core, xml, perf):
    cfg = {"PERFORMANCE_HINT": "LATENCY", "PERF_COUNT": perf}
    return core.compile_model(str(xml), "CPU", cfg)


def time_runs(compiled, x, n):
    req = compiled.create_infer_request()
    for _ in range(WARMUP):
        req.infer({0: x})
    ts, acc = [], defaultdict(float)
    info0 = None
    for _ in range(n):
        t = time.perf_counter()
        req.infer({0: x})
        ts.append((time.perf_counter() - t) * 1e3)
        if compiled.get_property("PERF_COUNT"):
            info = req.profiling_info
            info0 = info0 or info
            for p in info:
                acc[p.node_name] += p.real_time.total_seconds() * 1e3
    return statistics.median(ts), acc, info0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    core = ov.Core()
    rng = np.random.default_rng(0)
    x = rng.random(SHAPE, dtype=np.float32)
    lines = [f"# U-Net R34/512 OpenVINO CPU PERF_COUNT({time.strftime('%F %T')})", "",
             f"- 硬體: {core.get_property('CPU', 'FULL_DEVICE_NAME')};OpenVINO {ov.get_version()}",
             f"- 輸入: 合成影像 {SHAPE}(均勻亂數,種子 0);LATENCY hint;暖機 {WARMUP}、每輪 {RUNS} 次、"
             f"PERF_COUNT 關 / 開交替 {ROUNDS} 輪", ""]
    for tag, xml in MODELS.items():
        src = core.read_model(str(xml))
        src_types = Counter(op.get_type_name() for op in src.get_ops())
        lat = {False: [], True: []}
        acc_all = defaultdict(float)
        for _ in range(ROUNDS):
            for perf in (False, True):
                c = compile_model(core, xml, perf)
                med, acc, info0 = time_runs(c, x, RUNS)
                lat[perf].append(med)
                if perf:
                    for k, v in acc.items():
                        acc_all[k] += v / (RUNS * ROUNDS)
                    rt = c.get_runtime_model()
                    last_c, last_info = c, info0
        # 執行圖: 每個節點融合了哪些原始運算
        rows = []
        info_by = {p.node_name: p for p in last_info}
        for op in last_c.get_runtime_model().get_ops():
            rt_info = op.get_rt_info()
            name = op.get_friendly_name()
            orig = rt_info["originalLayersNames"].astype(str) if "originalLayersNames" in rt_info else ""
            ltype = rt_info["layerType"].astype(str) if "layerType" in rt_info else op.get_type_name()
            p = info_by.get(name)
            rows.append(dict(node=name, layer_type=ltype,
                             exec_type=p.exec_type if p else "",
                             status=str(p.status).split(".")[-1] if p else "",
                             ms=round(acc_all.get(name, 0.0), 4),
                             n_fused=len([s for s in orig.split(",") if s]),
                             original=orig))
        with open(OUT / f"nodes_{tag}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: -r["ms"]))
        executed = [r for r in rows if r["ms"] > 0]
        total = sum(r["ms"] for r in executed)
        # 卷積節點融合了哪些原始運算種類
        orig_type = {op.get_friendly_name(): op.get_type_name() for op in src.get_ops()}
        conv_rows = [r for r in rows if r["layer_type"] == "Convolution"]
        fused_kinds = Counter()
        for r in conv_rows:
            kinds = sorted({orig_type.get(s.strip(), "?") for s in r["original"].split(",") if s.strip()})
            fused_kinds[" + ".join(kinds)] += 1
        by_type = defaultdict(lambda: [0, 0.0])
        for r in executed:
            by_type[r["layer_type"]][0] += 1
            by_type[r["layer_type"]][1] += r["ms"]
        lines += [f"## {tag}(`{xml.relative_to(ROOT)}`)", "",
                  f"- 原始圖 {sum(src_types.values())} 個運算(Convolution {src_types['Convolution']}、"
                  f"FakeQuantize {src_types.get('FakeQuantize', 0)}、Add {src_types.get('Add', 0)}、"
                  f"Multiply {src_types.get('Multiply', 0)}、Relu {src_types.get('Relu', 0)})"
                  f" → 執行圖 {len(rows)} 個節點,實際有花時間的 {len(executed)} 個",
                  f"- 延遲中位數: PERF_COUNT 關 {', '.join(f'{v:.1f}' for v in lat[False])} ms;"
                  f"開 {', '.join(f'{v:.1f}' for v in lat[True])} ms;逐節點加總 {total:.1f} ms",
                  "", "卷積節點各融合了哪些原始運算(節點數):", ""]
        lines += [f"- {k}: {v}" for k, v in fused_kinds.most_common()]
        lines += ["", "| 節點類型 | 個數 | ms | 佔比 |", "| --- | --- | --- | --- |"]
        for t, (n, ms) in sorted(by_type.items(), key=lambda kv: -kv[1][1]):
            lines.append(f"| {t} | {n} | {ms:.2f} | {ms / total:.1%} |")
        lines += ["", "最花時間的 10 個節點:", "", "| 節點 | 類型 | exec_type | ms | 佔比 | 融合原始運算數 |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for r in sorted(executed, key=lambda r: -r["ms"])[:10]:
            lines.append(f"| {r['node']} | {r['layer_type']} | {r['exec_type']} | {r['ms']:.2f} | "
                         f"{r['ms'] / total:.1%} | {r['n_fused']} |")
        lines.append("")
        print(f"{tag}: done, total {total:.1f} ms")
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
    import os
    os._exit(0)  # OpenVINO 在直譯器關閉時會 access violation(結果已寫出),直接結束

