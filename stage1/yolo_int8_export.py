"""2026-11-13 兩個牙齒模型(第一階段 train_success_replica3、32 類 tufts_fdi_all_yolov8s-seg)匯出 OpenVINO INT8(使用者要求)。

- 匯出: ultralytics export(format="openvino", int8=True, dynamic=True, imgsz=1024),內部是 NNCF PTQ;與部署用的 FP32 匯出同樣是動態輸入。
- 校正資料(只用影像,不需要標註):
  - 第一階段: 原始訓練資料 yaml/20251217_alc_66.yaml 已不在伺服器(見待辦「第一階段 B」),改用 dental_project/original_pictures 中
    不屬於 kfold_pano 測試集的全景片,隨機 300 張(seed 0),連結到 compare_results/deploy_eval/yolo_int8/calib_teeth/images/。
  - 32 類: Tufts yolo_fdi_all 的 train 分割(與訓練時相同的 data.yaml),split="train"。
- 輸出: compare_results/deploy_eval/yolo_int8/{teeth,teeth32}_int8_openvino_model/(從 weights/ 底下搬過來,避免和共用的匯出混用)。
"""
import random
import shutil

import yaml
from ultralytics import YOLO

from stage2.compare_yolo_unet import ROOT
from deployment.pipeline_timing import PANO_DIR, TEETH_MODEL, test_panos
from stage1.teeth32_compare import WEIGHTS as TEETH32

OUT = ROOT / "compare_results" / "deploy_eval" / "yolo_int8"


def calib_yaml(n=300):
    test = {p.name for p in test_panos()}
    pool = sorted(p for p in PANO_DIR.iterdir() if p.suffix.lower() in (".jpg", ".png") and p.name not in test)
    pick = random.Random(0).sample(pool, min(n, len(pool)))
    img = OUT / "calib_teeth" / "images"
    img.mkdir(parents=True, exist_ok=True)
    for p in pick:
        link = img / p.name
        if not link.exists():
            link.symlink_to(p)
    names = YOLO(TEETH_MODEL).names
    path = OUT / "calib_teeth" / "data.yaml"
    path.write_text(yaml.safe_dump({"path": str(OUT / "calib_teeth"), "train": "images", "val": "images", "names": names}, allow_unicode=True))
    print(f"第一階段校正: 非測試全景片 {len(pick)} / {len(pool)} 張")
    return path


def export(weights, data, name, **kw):
    dst = OUT / f"{name}_int8_openvino_model"
    if dst.exists():
        print("已存在:", dst)
        return
    out = YOLO(weights).export(format="openvino", int8=True, dynamic=True, imgsz=1024, data=str(data), **kw)
    shutil.move(out, dst)
    print("匯出:", dst)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    export(TEETH_MODEL, calib_yaml(), "teeth", fraction=1.0)
    export(TEETH32, ROOT / "external/tufts/yolo_fdi_all/data.yaml", "teeth32", split="train", fraction=0.3)


if __name__ == "__main__":
    main()
