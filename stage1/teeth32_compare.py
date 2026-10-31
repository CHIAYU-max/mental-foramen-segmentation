import argparse
import csv
import pickle
import random
import sys
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO

from stage2.compare_yolo_unet import ROOT
from stage1.fdi_cropped import assign_fdi_indices
from stage1.tufts_teeth import FDI

# 32 類 FDI 牙齒模型(Tufts 訓練)在我們的全景片上和目前方法比較(2026-10-29,GPU 推論一次):
#   我們沒有 FDI 的 GT,所以只能看兩者一不一致,再由人目視判斷誰對。
#   目前方法: compare_results/fdi_v2/lower_teeth.pkl 的牙齒(目前牙齒模型)+ fdi_cropped.assign_fdi_indices。
#   新方法: runs/segment/tufts_fdi_all_yolov8s-seg(imgsz 1024),每個類別取信心最高的實例。
#   比對: 對每側的 x4 / x5 / x6,新方法該編號的外框中心落在目前方法的哪一顆牙(輪廓)內:
#     same   = 同一顆牙;shift±k = 落在目前方法編號 x(k 差)的牙;outside = 不在任何目前方法偵測到的牙內;
#     new_missing / cur_missing = 只有一方有這個編號。
#   疊圖: 依「該側 x4–x6 是否一致」分層隨機抽(--n-diff 張不一致、--n-same 張一致),左右兩半: 上 = 目前方法(紅)、下 = 新方法(綠)。
#   輸出 compare_results/teeth32/(per_tooth.csv、summary.txt、overlays/*.jpg、review.csv: 給使用者填「哪個對」)。
#   疊圖是病人影像,只放在伺服器上。
PANO_DIR = ROOT / "dental_project" / "20260826__mf_851"
CACHE = ROOT / "compare_results" / "fdi_v2" / "lower_teeth.pkl"
WEIGHTS = ROOT / "runs" / "segment" / "tufts_fdi_all_yolov8s-seg" / "weights" / "best.pt"
OUT = ROOT / "compare_results" / "teeth32"
TARGETS = ("4", "5", "6")
OVERLAY_WIDTH = 1400


def new_teeth(model, img):
    r = model.predict(img, imgsz=1024, conf=0.25, verbose=False)[0]
    best = {}
    for box, cls, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()):
        if cls not in best or conf > best[cls][1]:
            best[cls] = (box, float(conf))
    return {FDI[c]: (b, conf) for c, (b, conf) in best.items()}


def compare(cur, new):
    rows = []
    cur_by_id = {t["fdi_id"]: t for t in cur if t.get("fdi_id")}
    for quad in "43":
        for n in TARGETS:
            fdi = quad + n
            c, nw = cur_by_id.get(fdi), new.get(fdi)
            if c is None and nw is None:
                continue
            if nw is None:
                status = "new_missing"
            elif c is None:
                status = "cur_missing"
            else:
                x1, y1, x2, y2 = nw[0]
                center = (float((x1 + x2) / 2), float((y1 + y2) / 2))
                hit = next((t for t in cur if t.get("fdi_id") and cv2.pointPolygonTest(t["contour"].astype(np.int32), center, False) >= 0), None)
                if hit is None:
                    status = "outside"
                elif hit["fdi_id"] == fdi:
                    status = "same"
                elif hit["fdi_id"][0] == quad:
                    status = f"shift{int(hit['fdi_id'][1]) - int(n):+d}"
                else:
                    status = "other_side"
            rows.append({"fdi": fdi, "status": status, "new_conf": round(nw[1], 3) if nw else ""})
    return rows


def draw(img, cur, new, path):
    h, w = img.shape[:2]
    s = OVERLAY_WIDTH / w
    small = cv2.resize(img, (OVERLAY_WIDTH, int(h * s)))
    top, bottom = small.copy(), small.copy()
    for t in cur:
        if t.get("fdi_id", "")[1:] in TARGETS:
            cv2.polylines(top, [(t["contour"] * s).astype(np.int32)], True, (0, 0, 255), 2)
            cv2.putText(top, t["fdi_id"], (int(t["cx"] * s) - 12, int(t["cy"] * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    for fdi, (b, _) in new.items():
        if fdi[1:] in TARGETS and fdi[0] in "34":
            x1, y1, x2, y2 = (b * s).astype(int)
            cv2.rectangle(bottom, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(bottom, fdi, ((x1 + x2) // 2 - 12, (y1 + y2) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
    for im, label, color in ((top, "current (red)", (0, 0, 255)), (bottom, "32-class (green)", (0, 200, 0))):
        cv2.putText(im, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    # 只留下顎附近(影像下 2/3)以縮小檔案
    crop = slice(int(small.shape[0] * 0.35), small.shape[0])
    cv2.imwrite(str(path), np.vstack([top[crop], bottom[crop]]), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-diff", type=int, default=20)
    parser.add_argument("--n-same", type=int, default=10)
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    (OUT / "overlays").mkdir(parents=True, exist_ok=True)
    cache = pickle.loads(CACHE.read_bytes())
    model = YOLO(WEIGHTS)
    rows, per_pano = [], {}
    for name, c in sorted(cache.items()):
        img = cv2.imread(str(PANO_DIR / name))
        cur = assign_fdi_indices([dict(t) for t in sorted(c["lower"], key=lambda t: t["cx"])]) if c["lower"] else []
        new = new_teeth(model, img)
        comp = compare(cur, new)
        rows += [{"pano": name, **r} for r in comp]
        per_pano[name] = (cur, new, comp)

    status = Counter(r["status"] for r in rows)
    sides = Counter()
    for name, (_, _, comp) in per_pano.items():
        for quad in "43":
            st = [r["status"] for r in comp if r["fdi"][0] == quad]
            if st:
                sides["all_same" if all(s == "same" for s in st) else "differ"] += 1
    lines = [f"全景片 {len(per_pano)} 張;x4–x6 比對 {len(rows)} 顆: " + ", ".join(f"{k} {v}({v / len(rows):.1%})" for k, v in status.most_common()),
             f"每側 x4–x6 完全一致 {sides['all_same']} / {sum(sides.values())}({sides['all_same'] / max(sum(sides.values()), 1):.1%})"]
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    with open(OUT / "per_tooth.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    differ = [n for n, (_, _, comp) in per_pano.items() if any(r["status"] != "same" for r in comp)]
    same = [n for n in per_pano if n not in differ and per_pano[n][2]]
    rng = random.Random(0)
    picks = [(n, "differ") for n in rng.sample(differ, min(args.n_diff, len(differ)))] + [(n, "same") for n in rng.sample(same, min(args.n_same, len(same)))]
    rng.shuffle(picks)  # 不讓看的人知道哪些是一致的
    review = []
    for i, (name, group) in enumerate(picks, 1):
        cur, new, comp = per_pano[name]
        draw(cv2.imread(str(PANO_DIR / name)), cur, new, OUT / "overlays" / f"{i:02d}.jpg")
        review.append({"no": f"{i:02d}", "pano": name, "group": group,
                       "differences": " ".join(f"{r['fdi']}:{r['status']}" for r in comp if r["status"] != "same")})
    # 判讀表不放分組與差異(避免看的人先知道答案),另存答案表 review_key.csv
    with open(OUT / "review_key.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(review[0]))
        writer.writeheader()
        writer.writerows(review)
    with open(OUT / "review.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["no", "right_side_correct(current/new/both/neither)", "left_side_correct(current/new/both/neither)", "note"])
        writer.writeheader()
        writer.writerows({"no": r["no"]} for r in review)
    print(f"疊圖 {len(review)} 張: {OUT / 'overlays'};判讀表 {OUT / 'review.csv'}")


if __name__ == "__main__":
    main()
