# 2026-11-05 中線規則 vs 現行規則 疊圖(病人影像只存在伺服器 compare_results/mf_position/midline/overlays/)
# 抽樣(seed 0): 兩規則判定不同 24 個 + 相同 6 個;--blank 只畫幾何(黑底,供檢查繪圖)
import csv, json, pickle, random, sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path.home() / "YOLOv8"
sys.path.insert(0, str(ROOT))
from position.mf_position import ITEMS_JSON, KEY_CSV, centroid
import position.mf_position_eval as ev
from deployment.pipeline_timing import PANO_DIR
from position.midline_teeth32 import OUT, tooth_geom

NAMES = {1: "1 (mesial to 4)", 2: "2 (under 4)", 3: "3 (4-5)", 4: "4 (under 5)", 5: "5 (5-6)", 6: "6 (under 6)"}
WIDTH = 900


def P(p, s, o):
    return tuple(int(round(v)) for v in (np.asarray(p) - o) * s)


def line_at_y(c, v, y):
    return c + v * ((y - c[1]) / v[1])


def main(blank=False):
    data = json.loads(ITEMS_JSON.read_text(encoding="utf-8"))
    key = {r["alias"]: r for r in csv.DictReader(open(KEY_CSV, newline=""))}
    cache = pickle.loads((OUT / "teeth32_masks.pkl").read_bytes())
    rows = [r for r in csv.DictReader(open(OUT / "midline_teeth32.csv", newline="")) if r["crown"] and r["cur_s1"] and r["mid_s1"]]
    rng = random.Random(0)
    diff = [r for r in rows if r["cur_s1"] != r["mid_s1"]]
    same = [r for r in rows if r["cur_s1"] == r["mid_s1"]]
    pick = rng.sample(diff, 24) + rng.sample(same, 6)
    rng.shuffle(pick)
    folder = OUT / ("overlays_blank" if blank else "overlays")
    folder.mkdir(exist_ok=True)
    out_rows = []
    for i, r in enumerate(pick, 1):
        alias = r["alias"]
        k, item = key[alias], data["items"][alias]
        q = str(k["quadrant"])
        distal = -1.0 if q == "4" else 1.0
        cx, cy = int(k["crop_x"]), int(k["crop_y"])
        mf = np.array(item["mf"], dtype=float) + (cx, cy)
        m = centroid(mf)
        rad = float(r["r"])
        teeth = {n: cache[k["pano"]][q + n] for n in "3456" if q + n in cache[k["pano"]]}
        pts = np.vstack([teeth[n] for n in "456"] + [mf])
        x0, y0 = pts.min(0) - 60
        x1, y1 = pts.max(0) + 60
        img = cv2.imread(str(PANO_DIR / k["pano"]))
        H, W = img.shape[:2]
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(W, int(x1)), min(H, int(y1))
        crop = img[y0:y1, x0:x1].copy()
        if blank:
            crop[:] = 40
        s = WIDTH / crop.shape[1]
        view = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        o = np.array([x0, y0], float)
        # 32 類牙齒輪廓(白)與根尖寬度往下延伸的範圍(洋紅)
        for n, poly in teeth.items():
            cv2.polylines(view, [np.array([P(p, s, o) for p in poly], np.int32)], True, (230, 230, 230), 1, cv2.LINE_AA)
            c, a, nn, wc, wa, oc, oa = tooth_geom(poly, distal)
            tip = c + a * ((poly - c) @ a).max()
            for e in (oa - wa, oa + wa):
                p0 = tip + nn * (e - oa) + nn * oa - nn * ((tip - c) @ nn)
                t_end = (m - p0) @ a + 2 * rad
                cv2.line(view, P(p0, s, o), P(p0 + a * t_end, s, o), (255, 0, 255), 1, cv2.LINE_AA)
        # 第一階段參考線(青)與中線(黃),畫到頦孔下方
        lines = ev.lines_from_item(item, cx, cy, "a")
        ytop = min(c[1] for c, _, _ in lines.values())
        ybot = m[1] + 3 * rad
        for n, (c, v, nn) in lines.items():
            cv2.line(view, P(line_at_y(c, v, ytop), s, o), P(line_at_y(c, v, ybot), s, o), (255, 255, 0), 2, cv2.LINE_AA)
            lp = np.array(P(line_at_y(c, v, ybot - 0.5 * rad), s, o)) + (6, 0)
            cv2.putText(view, "x" + n, tuple(int(t) for t in lp), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        for a_, b_ in (("4", "5"), ("5", "6")):
            if a_ in lines and b_ in lines:
                ys = np.linspace(ytop, ybot, 30)
                mid = [(line_at_y(*lines[a_][:2], y) + line_at_y(*lines[b_][:2], y)) / 2 for y in ys]
                for j in range(0, len(mid) - 1, 2):
                    cv2.line(view, P(mid[j], s, o), P(mid[j + 1], s, o), (0, 220, 255), 2, cv2.LINE_AA)
        # 頦孔(紅)與質心
        cv2.polylines(view, [np.array([P(p, s, o) for p in mf], np.int32)], True, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(view, P(m, s, o), 3, (0, 0, 255), -1)
        cur, mid_ = int(r["cur_s1"]), int(r["mid_s1"])
        head = np.full((96, view.shape[1], 3), 255, np.uint8)
        cv2.putText(head, "cyan = tooth axis (stage 1)   yellow dash = midline   white = 32-class teeth   magenta = apex width   red = MF",
                    (8, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1, cv2.LINE_AA)
        cv2.putText(head, f"{i:02d}  current: {NAMES[cur]}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 120, 0), 2)
        cv2.putText(head, f"    midline: {NAMES[mid_]}" + ("   (same)" if cur == mid_ else ""), (8, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 200), 2)
        cv2.imwrite(str(folder / f"{i:02d}.jpg"), np.vstack([head, view]))
        out_rows.append({"no": f"{i:02d}", "alias": alias, "set": r["set"], "current": cur, "midline": mid_,
                         "apex_ref": r["apex"], "crown_ref": r["crown"], "your_call(current/midline/neither)": "", "note": ""})
    with open(folder / "overlays.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print("wrote", len(out_rows), "to", folder)



if __name__ == "__main__":
    main(blank="--blank" in sys.argv)
