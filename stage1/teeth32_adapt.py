"""2026-11-14 讓 32 類 FDI 牙齒模型(Tufts 訓練)適應我們的片子(使用者要求;內部實驗,Tufts 授權未確認)

背景(flag_patterns.py): 「兩種編號不一致」旗標在測試集正常牙的 16 個誤報裡,13 個是 32 類模型在我們的片子上整排差一號(現行編號是對的)。
做法(事先固定,不看測試集調): 自我訓練。
  1. prepare: 偽標註來源 = dental_project/original_pictures 裡「沒有頦孔 GT」的全景片(不含測試集與非測試集標註清單的全景片;
     另外用縮圖相關係數 > 0.97 排除與它們近似重複的影像)。每張跑現行牙齒模型(依序編號)與 32 類模型;
     只留「左右兩側的 x4–x6 六顆都一致(same)」的影像(一致時目視 93% 正確),用 32 類模型自己的預測(每類信心最高、≥ 0.5)當整張的偽標註。
  2. train: 從 tufts_fdi_all 的權重接著訓練,資料 = Tufts 訓練集 + 偽標註;驗證 = Tufts 驗證集(看有沒有退步)。
     yolov8s-seg、imgsz 1024、batch 4、40 epochs、fliplr = 0、seed 0。輸出 runs/segment/tufts_fdi_adapt_yolov8s-seg。
  3. eval: 與 fdi_flag.py 相同的比對(現行編號取 items.json;32 類模型 PyTorch、imgsz 1024),比較原本與適應後的模型:
     測試集(複查後標註)正常牙被標記數、異常側被標記數;非測試集有人工標註(mf_position/dev_abnormal/labels_rater.csv)時也算;
     另報全部非測試集的側「x4–x6 全部一致」的比例(不需要標註)。
事先定的判準(測試集只看這一次): 正常牙誤標 ≤ 11 / 103(原本 16,至少少 5 側)且異常側被標記 ≥ 20 / 29(原本 21)→ 才進一步匯出 INT8、做部署評分;否則不採用。
輸出: compare_results/teeth32_adapt/(pseudo_manifest.csv、summary.md、per_side.csv)
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from stage2.compare_yolo_unet import ROOT
from stage1.fdi_cropped import assign_fdi_indices, process_and_separate_jaws
from position.fdi_flag import side_status
from position.mf_position import ITEMS_JSON, KEY_CSV, contour
from deployment.pipeline_timing import PANO_DIR, TEETH_MODEL
from stage1.stage1_eval import Sized
from stage1.teeth32_compare import WEIGHTS, compare, new_teeth
from stage1.tufts_teeth import FDI, TUFTS

OUT = ROOT / "compare_results" / "teeth32_adapt"
DATA = TUFTS / "yolo_fdi_adapt"
NAME = "tufts_fdi_adapt_yolov8s-seg"
ADAPTED = ROOT / "runs" / "segment" / NAME / "weights" / "best.pt"
DEV_LABELS = ROOT / "mf_position" / "dev_abnormal" / "labels_rater.csv"
ADJ = ROOT / "compare_results" / "mf_position" / "labels_adjudicated_dir" / "labels_rater.csv"
MIN_CONF, DUP = 0.5, 0.97


def thumb(path):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    t = cv2.resize(g, (64, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    return (t - t.mean()) / (t.std() + 1e-6) / np.sqrt(t.size)


def prepare(args):
    OUT.mkdir(parents=True, exist_ok=True)
    keys = list(csv.DictReader(open(KEY_CSV, newline="")))
    labeled = {k["pano"] for k in keys}  # 測試集 + 非測試集標註清單(都有頦孔 GT)
    files = sorted(p for p in PANO_DIR.rglob("*") if p.suffix.lower() in (".jpg", ".png"))
    rel = lambda p: p.relative_to(PANO_DIR).as_posix()
    held = np.stack([t for t in (thumb(PANO_DIR / n) for n in sorted(labeled)) if t is not None])
    pool, dropped = [], 0
    for p in files:
        if rel(p) in labeled:
            continue
        t = thumb(p)
        if t is None:
            continue
        if float((held @ t).max()) > DUP:
            dropped += 1
            continue
        pool.append(p)
    pool = pool[:args.limit or None]
    print(f"全景片 {len(files)} 張;有頦孔 GT(測試 + 標註清單){len(labeled)} 張不用;近似重複排除 {dropped} 張;偽標註候選 {len(pool)} 張", flush=True)
    teeth = Sized(YOLO(TEETH_MODEL), 1024)
    model = YOLO(WEIGHTS)
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            (DATA / split / sub).mkdir(parents=True, exist_ok=True)
    src = TUFTS / "yolo_fdi_all"
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            for f in (src / split / sub).iterdir():
                link = DATA / split / sub / f.name
                if not link.exists():
                    link.symlink_to(f.resolve())
    for f in (DATA / "train" / "images").glob("ours_*"):
        f.unlink()
    for f in (DATA / "train" / "labels").glob("ours_*"):
        f.unlink()
    manifest, kept = [], 0
    for i, p in enumerate(pool, 1):
        img = cv2.imread(str(p))
        _, lower = process_and_separate_jaws(img, teeth)
        cur = assign_fdi_indices(lower) if lower else []
        for t in cur:
            t["contour"] = contour(t)
        r = model.predict(img, imgsz=1024, conf=0.25, verbose=False)[0]
        best = {}
        if r.masks is not None:
            for j, (box, cls, conf) in enumerate(zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy())):
                if cls not in best or conf > best[cls][1]:
                    best[cls] = (box, float(conf), j)
        new = {FDI[c]: (b, conf) for c, (b, conf, _) in best.items()}
        comp = compare(cur, new)
        ok = len(comp) == 6 and all(x["status"] == "same" for x in comp)
        lines = []
        if ok:
            for c, (_, conf, j) in sorted(best.items()):
                poly = r.masks.xyn[j]
                if conf >= MIN_CONF and len(poly) >= 3:
                    lines.append(f"{c} " + " ".join(f"{v:.6f}" for v in poly.ravel()))
            ok = len(lines) >= 12
        if ok:
            name = f"ours_{kept:04d}"
            (DATA / "train" / "images" / f"{name}{p.suffix.lower()}").symlink_to(p.resolve())
            (DATA / "train" / "labels" / f"{name}.txt").write_text("\n".join(lines) + "\n")
            kept += 1
        manifest.append({"pano": rel(p), "used": int(ok), "instances": len(lines), "statuses": " ".join(f"{x['fdi']}:{x['status']}" for x in comp)})
        if i % 100 == 0:
            print(f"[{i}/{len(pool)}] 已採用 {kept}", flush=True)
    (DATA / "data.yaml").write_text(f"path: {DATA.as_posix()}\ntrain: train/images\nval: val/images\nnames:\n" + "".join(f"  {i}: '{n}'\n" for i, n in enumerate(FDI)))
    with open(OUT / "pseudo_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)
    print(f"偽標註: 候選 {len(pool)} 張,兩側 x4–x6 都一致且採用 {kept} 張;Tufts 訓練 {len(list((src / 'train' / 'images').iterdir()))} 張", flush=True)


def train(args):
    model = YOLO(WEIGHTS)
    model.train(data=str(DATA / "data.yaml"), imgsz=1024, batch=4, epochs=args.epochs, patience=20, fliplr=0.0, seed=0, workers=8, cache=False,
                fraction=args.fraction, project=str(ROOT / "runs" / "segment"), name=NAME + args.suffix, exist_ok=True, plots=False)
    (ROOT / "runs" / "segment" / (NAME + args.suffix) / "weights" / "last.pt").unlink(missing_ok=True)


def evaluate(args):
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads(ITEMS_JSON.read_text(encoding="utf-8"))
    key = {r["alias"]: r for r in csv.DictReader(open(KEY_CSV, newline=""))}
    test_panos = {k["pano"] for k in key.values() if k["set"] == "test"}
    grp = {}
    for r in csv.DictReader(open(ADJ, newline="", encoding="utf-8-sig")):
        g = "FDI 錯" if r["fdi_wrong"] == "1" else "缺牙" if r["position"] == "M" else "正常牙" if r["position"] in tuple("123456") else None
        if g and key.get(r["alias"], {}).get("set") == "test":
            grp[("test", r["alias"])] = g
    if DEV_LABELS.exists():
        for r in csv.DictReader(open(DEV_LABELS, newline="")):
            g = {"N": "正常牙", "M": "缺牙", "W": "FDI 錯"}.get(r["position"])
            if str(r.get("fdi_wrong")) == "1":
                g = "FDI 錯"
            if g:
                grp[("dev", r["alias"])] = g
    models = [("原本(tufts_fdi_all)", WEIGHTS)] + ([("適應後(tufts_fdi_adapt)", Path(args.weights) if args.weights else ADAPTED)] if (args.weights or ADAPTED.exists()) else [])
    aliases = [a for a in data["order"] if not args.limit or key[a]["pano"] in sorted({key[x]["pano"] for x in data["order"]})[:args.limit]]
    rows, lines = [], ["# 32 類模型適應我們的片子(teeth32_adapt.py eval)", ""]
    for name, weights in models:
        model = YOLO(weights)
        cache, flag = {}, {}
        for a in aliases:
            k, item = key[a], data["items"][a]
            if k["pano"] not in cache:
                cache[k["pano"]] = new_teeth(model, cv2.imread(str(PANO_DIR / k["pano"])))
            off = np.array([float(k["crop_x"]), float(k["crop_y"])])
            cur = [(t["fdi"], np.asarray(t["poly"], dtype=float) + off) for t in item["teeth"] if len(t["fdi"]) == 2]
            st = side_status(cur, cache[k["pano"]], str(k["quadrant"]))
            flag[a] = int(any(st[n] != "same" for n in "456"))
            rows.append({"model": name, "alias": a, "set": "test" if k["pano"] in test_panos else "dev", **{f"x{n}": st[n] for n in "456"}, "flag": flag[a]})
        lines.append(f"## {name}")
        for setting, label in (("test", "測試集(複查後標註)"), ("dev", "非測試集(人工標註)")):
            sub = {a: g for (s, a), g in grp.items() if s == setting and a in flag}
            if not sub:
                lines.append(f"- {label}: 沒有標註")
                continue
            c = Counter(g for g in sub.values())
            nm = [a for a, g in sub.items() if g == "正常牙"]
            ab = [a for a, g in sub.items() if g != "正常牙"]
            lines.append(f"- {label}(正常牙 {c['正常牙']}、缺牙 {c['缺牙']}、FDI 錯 {c['FDI 錯']}): 正常牙誤標 {sum(flag[a] for a in nm)} / {len(nm)};"
                         f"異常側被標記 {sum(flag[a] for a in ab)} / {len(ab)}(缺牙 {sum(flag[a] for a in ab if sub[a] == '缺牙')}、FDI 錯 {sum(flag[a] for a in ab if sub[a] == 'FDI 錯')})")
        dev_all = [a for a in flag if key[a]["pano"] not in test_panos]
        lines.append(f"- 非測試集全部 {len(dev_all)} 側: x4–x6 全部一致 {np.mean([1 - flag[a] for a in dev_all]):.1%}" if dev_all else "- 非測試集: 無")
        lines.append("")
        print(f"{name} 完成", flush=True)
    text = "\n".join(lines)
    (OUT / f"summary{args.suffix}.md").write_text(text + "\n", encoding="utf-8")
    with open(OUT / f"per_side{args.suffix}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(text)


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "train", "eval"])
    parser.add_argument("--limit", type=int, default=0, help="試跑: prepare 只處理前 N 張候選;eval 只看前 N 張全景片")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--suffix", default="", help="試跑時加在輸出名稱後面")
    parser.add_argument("--weights")
    args = parser.parse_args()
    {"prepare": prepare, "train": train, "eval": evaluate}[args.command](args)
