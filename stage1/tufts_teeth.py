import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from stage2.compare_yolo_unet import ROOT

# 用 Tufts 重新訓練牙齒模型,直接做 32 類 FDI(2026-10-28):
#   目前第一階段是「牙齒分割(單一類別)→ 從中線依序編號」,Tufts 上 x4–x6 只有約 70% 正確,錯誤多半是前面的牙齒沒偵測到,
#   後面全部少一號(fdi_tufts.py)。這裡讓模型直接分類每顆牙的 FDI 編號,不必依序數。
#   資料: Tufts teeth_polygon.json(Universal 編號 → FDI);只用恆牙的 918 張(有乳牙的 82 張排除)。
#     Tufts 一顆牙常拆成多塊碎片(第一次試跑每張約 105 個實例),合併成一顆牙一個輪廓(merge)。
#   分割(與 fdi_tufts.py 相同): 開發組 = 偶數編號(再分 90% 訓練 / 10% 驗證,seed 0),測試組 = 奇數編號。
#     --split dev: 用開發組訓練,在測試組評估(與目前版本比較);--split all: 全部恆牙(90% / 10%)訓練最終模型。
#   訓練: yolov8s-seg(與目前牙齒模型相同)、imgsz 1024、batch 4、150 epochs;**fliplr = 0**(左右翻轉會把 3x 與 4x 對調)。
#   評估(--eval): 測試組每張每個類別取信心最高的實例,看 x4–x6 的質心落在哪一顆 GT 牙齒內(與 fdi_tufts.evaluate 相同的指標)。
TUFTS = ROOT / "external" / "tufts"
FDI = [f"{q}{n}" for q in "1234" for n in range(1, 9)]
TARGETS = ("4", "5", "6")


def universal_to_fdi_all(title):
    # Universal 1–8 = 18–11、9–16 = 21–28、17–24 = 38–31、25–32 = 41–48;乳牙(A–T)回傳 None
    if not title.isdigit():
        return None
    u = int(title)
    if 1 <= u <= 8:
        return f"1{9 - u}"
    if 9 <= u <= 16:
        return f"2{u - 8}"
    if 17 <= u <= 24:
        return f"3{25 - u}"
    if 25 <= u <= 32:
        return f"4{u - 24}"
    return None


def load_entries():
    entries = {}
    for e in json.loads((TUFTS / "teeth_polygon.json").read_text()):
        stem = e["External ID"].rsplit(".", 1)[0]
        objs = e["Label"]["objects"]
        primary = any(len(o["title"]) == 1 and o["title"].isalpha() for o in objs)
        teeth = {}
        for o in objs:
            fdi = universal_to_fdi_all(o["title"])
            if fdi and o.get("polygons") not in (None, "none"):
                polys = json.loads(o["polygons"]) if isinstance(o["polygons"], str) else o["polygons"]
                merged = merge([np.array(p, dtype=np.float32) for p in polys if len(p) >= 3])
                if merged is not None:
                    teeth[fdi] = [merged]
        entries[stem] = {"primary": primary, "teeth": teeth}
    return entries


def merge(polys):
    # Tufts 常把一顆牙的多邊形拆成好幾塊碎片;畫成遮罩、閉運算補縫,取最大的外輪廓當這顆牙(一顆牙一個實例)
    if not polys:
        return None
    x1, y1 = np.floor(np.min([p.min(0) for p in polys], axis=0)).astype(int) - 5
    x2, y2 = np.ceil(np.max([p.max(0) for p in polys], axis=0)).astype(int) + 5
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    for p in polys:
        cv2.fillPoly(mask, [np.round(p - (x1, y1)).astype(np.int32)], 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea) if cs else None
    return None if c is None or cv2.contourArea(c) < 50 else (c.reshape(-1, 2) + (x1, y1)).astype(np.float32)


def splits(entries, mode):
    perm = sorted((s for s, e in entries.items() if not e["primary"]), key=int)
    test = [s for s in perm if int(s) % 2 == 1]
    pool = [s for s in perm if int(s) % 2 == 0] if mode == "dev" else perm
    pool = pool[:]
    random.Random(0).shuffle(pool)
    n_val = len(pool) // 10
    return {"train": sorted(pool[n_val:], key=int), "val": sorted(pool[:n_val], key=int), "test": test}


def build(entries, mode):
    root = TUFTS / f"yolo_fdi_{mode}"
    for split, stems in splits(entries, mode).items():
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        for stem in stems:
            src = next((TUFTS / "radiographs").glob(f"{stem}.*"))
            link = root / split / "images" / src.name
            if not link.exists():
                try:
                    link.symlink_to(src.resolve())
                except OSError:  # Windows 沒有權限建立連結時改成複製
                    link.write_bytes(src.read_bytes())
            h, w = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE).shape
            lines = [f"{FDI.index(fdi)} " + " ".join(f"{v:.6f}" for v in (p / (w, h)).ravel())
                     for fdi, polys in entries[stem]["teeth"].items() for p in polys]
            (root / split / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    yaml = root / "data.yaml"
    yaml.write_text(f"path: {root.as_posix()}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n"
                    + "".join(f"  {i}: '{n}'\n" for i, n in enumerate(FDI)))
    return yaml


def gt_of(point, teeth):
    for fdi, polys in teeth.items():
        if any(cv2.pointPolygonTest(p, point, False) >= 0 for p in polys):
            return fdi
    return None


def evaluate(weights, entries, mode, imgsz):
    model = YOLO(weights)
    stems = splits(entries, mode)["test"]
    teeth_n = teeth_ok = sides_all = sides_has = 0
    per = {k: [0, 0] for k in TARGETS}
    for stem in stems:
        src = next((TUFTS / "radiographs").glob(f"{stem}.*"))
        r = model.predict(str(src), imgsz=imgsz, conf=0.25, verbose=False)[0]
        best = {}
        for box, cls, conf in zip(r.boxes.xywh.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()):
            if cls not in best or conf > best[cls][1]:
                best[cls] = (box, conf)
        gt = entries[stem]["teeth"]
        for quad in "43":
            pred = {FDI[c]: b for c, (b, _) in best.items() if FDI[c][0] == quad and FDI[c][1] in TARGETS}
            sides_has += bool(pred)
            ok = 0
            for fdi, b in pred.items():
                g = gt_of((float(b[0]), float(b[1])), gt)
                teeth_n += 1
                ok += g == fdi
                per[fdi[1]][0] += g == fdi
                per[fdi[1]][1] += 1
            teeth_ok += ok
            gt_targets = [quad + n for n in TARGETS if quad + n in gt]
            sides_all += bool(gt_targets) and ok == len(pred) and set(pred) == set(gt_targets)
    sides = 2 * len(stems)
    print(f"測試組 {len(stems)} 張: x4–x6 牙齒正確 {teeth_ok / max(teeth_n, 1):.4f}({teeth_ok} / {teeth_n})、"
          f"每側全對 {sides_all / sides:.4f}、每側有 x4–x6 {sides_has / sides:.4f};"
          + "、".join(f"x{k} {a / max(b, 1):.3f}({b})" for k, (a, b) in per.items()))
    print("對照(fdi_tufts.py,測試組,GT 合併碎片): 目前版本 0.6796 / 每側全對 0.4968 / 有 0.8556;大臼齒錨點 0.7128 / 0.5237 / 0.8879")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "all"], default="dev")
    parser.add_argument("--model", default="yolov8s-seg.pt")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--workers", type=int, default=8, help="資料載入程序數;2026-10-28 筆電(16 GB RAM)用預設 8 時記憶體不足被中止,改用 2")
    parser.add_argument("--eval", default=None, help="只評估這個權重(測試組)")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    entries = load_entries()
    if args.eval:
        evaluate(args.eval, entries, args.split, args.imgsz)
        return
    yaml = build(entries, args.split)
    name = f"tufts_fdi_{args.split}_{Path(args.model).stem}"
    model = YOLO(args.model)
    model.train(data=str(yaml), imgsz=args.imgsz, batch=args.batch, epochs=args.epochs, patience=50, fliplr=0.0, seed=0, workers=args.workers, cache=False,
                project=str(ROOT / "runs" / "segment"), name=name, exist_ok=True, plots=True)
    best = ROOT / "runs" / "segment" / name / "weights" / "best.pt"
    (best.parent / "last.pt").unlink(missing_ok=True)
    if args.split == "dev":  # all 的訓練資料包含測試組,不評估
        evaluate(best, entries, "dev", args.imgsz)


if __name__ == "__main__":
    main()
