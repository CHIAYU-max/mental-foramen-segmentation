"""2026-11-15 長時間負載對照的彙整(sustained.ps1 每次跑完自動呼叫)。
每個 Tag: 依經過時間分 0–30 / 30–60 / 60–120 / 120–180 秒,報第一階段與 U-Net INT8 的中位數 ms、該段 CPU 實際效能 / 溫度 / 獨顯功耗平均。
11/15 加: 模式(--mode,舊的四次 = any)與 P / E 核心使用率(每段平均;U-Net 4 個執行緒全在 P 核心 ≈ P 50%、E 0%;第一階段 any 模式本來就會用到 E 核心)。
輸出: compare_results/deploy_eval/sustained/report.md
"""
import csv
import statistics
from pathlib import Path

DIR = Path(__file__).resolve().parents[2] / "compare_results" / "deploy_eval" / "sustained"
BINS = [(0, 30), (30, 60), (60, 120), (120, 180)]
lines = ["# 長時間負載對照(每段: 第一階段 / U-Net INT8 中位數 ms;效能 %、溫度 °C、獨顯 W、P / E 核心使用率為該段平均)", "",
         "| Tag | 模式 | 螢幕 | " + " | ".join(f"{a}–{b} 秒" for a, b in BINS) + " |", "| --- | --- | --- | " + " | ".join("---" for _ in BINS) + " |"]
for calls in sorted(DIR.glob("*_calls.csv"), key=lambda p: p.stat().st_mtime):
    tag = calls.name[:-len("_calls.csv")]
    c = list(csv.DictReader(open(calls, newline="")))
    sp = DIR / f"{tag}_sensors.csv"
    s = list(csv.DictReader(open(sp, newline="", encoding="utf-8-sig"))) if sp.exists() else []
    if not c:
        continue
    e0 = float(c[0]["epoch"]) - float(c[0]["t_s"])
    cells = []
    for a, b in BINS:
        cc = [r for r in c if a <= float(r["t_s"]) < b]
        ss = [r for r in s if a <= float(r["epoch"]) - e0 < b]
        if not cc:
            cells.append("")
            continue
        m = lambda name: statistics.median(float(r["ms"]) for r in cc if r["model"] == name)
        avg = lambda k: statistics.mean(float(r[k]) for r in ss if r[k] not in ("", None)) if any(r[k] not in ("", None) for r in ss) else float("nan")
        cells.append(f"{m('stage1'):.0f} / {m('unet_int8'):.1f}<br>{avg('perf_pct'):.0f}%、{avg('temp_c'):.0f}°C、{avg('dgpu_power_w'):.1f} W"
                     + (f"<br>P {avg('p_util'):.0f}% / E {avg('e_util'):.0f}%" if s and "p_util" in s[0] else ""))
    scr = sorted({r["screens"] for r in s}) or ["?"]
    mode = s[0].get("mode", "any") if s else "any"
    lines.append(f"| {tag} | {mode} | {'/'.join(scr)} | " + " | ".join(cells) + " |")
text = "\n".join(lines) + "\n"
(DIR / "report.md").write_text(text, encoding="utf-8")
print(text)
