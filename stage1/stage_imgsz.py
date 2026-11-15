"""2026-11-13 兩個牙齒模型降低輸入解析度的開發集檢查(使用者要求「都幫我直接做」)。

- 開發集同 stage1_int8.py: 非測試集、有頦孔 GT 的全景片 303 張(476 個頦孔)。
- 第一階段(OpenVINO FP32,動態輸入): imgsz 896 / 768 / 640 與 1024 的牙位一致率(GT 頦孔 + 參考線,center 規則)。
- 32 類模型(OpenVINO INT8,只用在旗標): 第一階段固定 FP32 @ 1024,32 類模型 imgsz 896 / 768 / 640 與 1024 的
  「兩種編號不一致」旗標相同的比例(每張全景片兩側)。
- 事先定的判準: 第一階段一致率 ≥ 0.99、旗標相同 ≥ 0.97 才進測試集(deploy_pipeline.py --teeth-imgsz / --t32-imgsz,5 種子,
  與 score_mixed_union 配對: 牙位差 ≥ −1 點、轉介率差在 ±1 點內),且筆電每張至少省 50 ms 才考慮採用。
輸出: compare_results/deploy_eval/imgsz/results.csv
"""
import csv
import json

import cv2
import numpy as np
from ultralytics import YOLO

import stage1.stage1_int8 as S
import stage1.teeth32_compare as teeth32_compare
from stage2.compare_yolo_unet import ROOT
from position.fdi_flag import side_status
from position.mf_position import contour
from deployment.pipeline_timing import PANO_DIR

OUT = ROOT / "compare_results" / "deploy_eval" / "imgsz"
T32 = ROOT / "compare_results" / "deploy_eval" / "yolo_int8" / "teeth32_int8_openvino_model"
SIZES = (896, 768, 640)


class Sized1(S.Stage1):
    imgsz = 1024

    def __call__(self, *args, **kwargs):
        return self.yolo(*args, imgsz=self.imgsz, device="intel:cpu", **kwargs)


class T32Sized:
    def __init__(self):
        self.model = YOLO(T32, task="segment")
        self.imgsz = 1024

    def predict(self, img, imgsz=1024, **kwargs):
        return self.model.predict(img, imgsz=self.imgsz, device="intel:cpu", **kwargs)


def log(row):
    path = OUT / "results.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "imgsz", "agree", "n", "changed", "shift_mean", "shift_p95", "note"])
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in w.fieldnames})
    print(row, flush=True)


def flags(lower, new):
    out = []
    for q in (3, 4):
        cur = [(t["fdi_id"], contour(t)) for t in lower if t.get("quadrant") == q and len(t.get("fdi_id", "")) == 2]
        st = side_status(cur, new, str(q))
        out.append(int(any(st[n] != "same" for n in "456")))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ref = S.build_ref(S.dev_set())
    s1 = Sized1()
    for size in SIZES:
        s1.imgsz = size
        r, _ = S.evaluate(s1, ref)
        log({"model": "stage1 FP32", "imgsz": size, **r})
    s1.imgsz = 1024
    t32 = T32Sized()
    base, cand = {}, {size: {} for size in SIZES}
    for i, pano in enumerate(ref, 1):
        img, lower = S.run_pano(s1, pano)
        t32.imgsz = 1024
        base[pano] = flags(lower, teeth32_compare.new_teeth(t32, img))
        for size in SIZES:
            t32.imgsz = size
            cand[size][pano] = flags(lower, teeth32_compare.new_teeth(t32, img))
        if i % 50 == 0:
            print(f"flags {i}/{len(ref)}", flush=True)
    n = 2 * len(base)
    flagged = sum(sum(v) for v in base.values())
    for size in SIZES:
        same = sum(a == b for p in base for a, b in zip(base[p], cand[size][p]))
        log({"model": "teeth32 INT8 (flags)", "imgsz": size, "agree": same / n, "n": n, "changed": n - same,
             "note": f"flagged at 1024: {flagged}; at {size}: {sum(sum(v) for v in cand[size].values())}"})
    (OUT / "flags.json").write_text(json.dumps({"base": base, **{str(k): v for k, v in cand.items()}}))


if __name__ == "__main__":
    main()
