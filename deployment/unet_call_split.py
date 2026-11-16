"""2026-11-14 第二階段每次呼叫 U-Net 的時間拆分(筆電 CPU;用合成影像,不需要病人影像)。

deploy_pipeline.Int8UNet.__call__ = 縮放到 512 → OpenVINO INT8 推論 → logits 雙線性放大回 ROI 尺寸 → argmax → 轉成遮罩。
量每一段的時間,看推論以外的前後處理佔多少。ROI 尺寸取測試集常見的大小(約 400 × 520)與較大的(600 × 800)。
輸出: compare_results/deploy_eval/unet_call_split_<主機>/summary.md
"""
import deployment.no_throttle as no_throttle  # 2026-11-16: 背景執行時 Windows 節能排程會讓第一階段慢約 1.5 倍
no_throttle.disable()
import platform
import statistics
import time

import numpy as np
import openvino as ov
import torch

from stage2.compare_yolo_unet import NAMES, ROOT
from deployment.int8_quant import unet_input

SIZE = (512, 512)
XML = ROOT / "compare_results" / "int8_pano" / "models" / "s0_fold0_int8.xml"
OUT = ROOT / "compare_results" / "deploy_eval" / f"unet_call_split_{platform.node()}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    compiled = ov.Core().compile_model(str(XML), "CPU")
    rng = np.random.default_rng(0)
    lines = [f"# U-Net 每次呼叫的時間拆分({platform.node()},INT8,50 次取中位數 ms)", "", "| ROI 尺寸 | 縮放到 512 | OpenVINO 推論 | logits 放大回 ROI | argmax | 轉成遮罩 | 合計 | 推論以外佔比 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    extra = []
    for h, w in ((400, 520), (600, 800)):
        gray = rng.integers(0, 255, (h, w), dtype=np.uint8)
        t = {k: [] for k in ("resize", "infer", "upsample", "argmax", "masks")}
        for i in range(60):
            a = time.perf_counter()
            x = unet_input(gray, SIZE)
            b = time.perf_counter()
            y = compiled(x)[0]
            c = time.perf_counter()
            up = torch.nn.functional.interpolate(torch.from_numpy(y), size=(h, w), mode="bilinear", align_corners=False)
            d = time.perf_counter()
            pred = up.argmax(1)[0].numpy()
            e = time.perf_counter()
            masks = {cls: pred == k + 1 for k, cls in NAMES.items()}
            f = time.perf_counter()
            if i >= 10:
                for k, v in zip(t, (b - a, c - b, d - c, e - d, f - e)):
                    t[k].append(v * 1e3)
        fast, same = [], True
        for i in range(60):
            y = compiled(unet_input(gray, SIZE))[0]
            up = torch.nn.functional.interpolate(torch.from_numpy(y), size=(h, w), mode="bilinear", align_corners=False)
            d = time.perf_counter()
            l0, l1, l2 = up[0].numpy()
            new = {NAMES[0]: (l1 > l0) & (l1 >= l2), NAMES[1]: (l2 > l0) & (l2 > l1)}
            e = time.perf_counter()
            pred = up.argmax(1)[0].numpy()
            same &= all((new[cls] == (pred == k + 1)).all() for k, cls in NAMES.items())
            if i >= 10:
                fast.append((e - d) * 1e3)
        # 平手的情況: 把 logits 量化成整數,製造大量平手再比一次
        q = torch.round(up * 2)
        l0, l1, l2 = q[0].numpy()
        pq = q.argmax(1)[0].numpy()
        ties = bool(((l1 > l0) & (l1 >= l2) == (pq == 1)).all() and ((l2 > l0) & (l2 > l1) == (pq == 2)).all())
        extra.append(f"- {h} × {w}: 直接比大小 {statistics.median(fast):.1f} ms(原本 argmax + 轉遮罩 {statistics.median(t['argmax']) + statistics.median(t['masks']):.1f} ms);"
                     f"輸出逐像素相同 {same};大量平手時也相同 {ties}")
        m = {k: statistics.median(v) for k, v in t.items()}
        tot = sum(m.values())
        lines.append(f"| {h} × {w} | {m['resize']:.1f} | {m['infer']:.1f} | {m['upsample']:.1f} | {m['argmax']:.1f} | {m['masks']:.1f} | {tot:.1f} | {1 - m['infer'] / tot:.0%} |")
    text = "\n".join(lines + ["", "## 快速版本(--fast-argmax)", ""] + extra)
    (OUT / "summary.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
