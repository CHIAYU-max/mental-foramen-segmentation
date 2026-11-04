"""2026-11-04 牙齒參考線改版的比較(CPU,不重跑牙齒模型、不改 mf_position/)。

問題(使用者發現 35 的參考線沒有沿著牙根): mf_position.reference_lines 的保險規則
  - x4 / x5 方向差 > 15° 時「相信比較接近垂直的那一顆」,把另一顆換成它的方向(x5 23%)。全景片上小臼齒本來就常往遠心傾斜,
    牙根清楚時傾斜是真的,這個規則會把正確的長軸換掉。
  - x6 用兩個牙根一起算主成分,兩根前後分開時主方向接近水平,79% 被換成 x5 的方向。
v2(門檻事先固定,不依結果調整):
  - 小臼齒: 一律用自己的牙根主成分;只在遮罩不可靠時借用相鄰牙的方向:
      牙根像素 < 20,或細長度 sqrt(λ1/λ2) < ELONG_MIN(形狀不像一條牙根);
      或與相鄰牙差 > MAX_DIFF_V2(明顯錯誤,例如遮罩黏到隔壁),此時換掉細長度較低的那一顆。
  - x6: 用 x4 / x5 的平均方向切成近心 / 遠心兩半,兩個牙根各自取下半部算主成分,方向取兩者平均,線通過兩根像素的中心;
      一根不可靠時只用另一根;兩根都不可靠,或結果與 x5 差 > MAX_DIFF_V2,改用 x5 的方向。
資料: mf_position/items.json 的牙齒輪廓(prepare 時的第一階段結果與 FDI,approxPolyDP 1.5 px)填回遮罩;
  先用現行的 reference_lines 重算,核對與 key.csv 的判定相同,確認重建的遮罩可用。
輸出: compare_results/mf_position/axis_v2/(compare.csv、summary.txt、overlays/ 病人影像只在伺服器)
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter

import cv2
import numpy as np

from stage2.compare_yolo_unet import ROOT
from position.mf_position import IMAGES, ITEMS_JSON, KEY_CSV, angle, classify, distal_normal, principal
from position.mf_position import reference_lines_v1 as reference_lines  # 現行版(2026-11-04 起改名為 v1)

OUT = ROOT / "compare_results" / "mf_position" / "axis_v2"
ELONG_MIN = 1.5
MAX_DIFF_V2 = 35.0
MIN_PX = 20
PAD = 800


def elongation(pts):
    lam = np.linalg.eigvalsh(np.cov((pts - pts.mean(axis=0)).T))
    return float(np.sqrt(lam[-1] / max(lam[0], 1e-9)))


def root_axis(pts, c0=None, v0=None):
    # 牙根 = 用整顆牙主成分定出的下半部;回傳 (中心, 方向, 細長度) 或 None
    if c0 is None:
        c0, v0 = principal(pts)
    root = pts[(pts - c0) @ v0 > 0]
    if len(root) < MIN_PX:
        return None
    c, v = principal(root)
    return c, v, elongation(root)


def reference_lines_v2(pts, quadrant, notes):
    # pts: {"4"/"5"/"6": 像素座標};回傳格式同 reference_lines
    axes, elong = {}, {}
    for k in "45":
        if k in pts:
            r = root_axis(pts[k])
            if r is None:
                c0, v0 = principal(pts[k])
                axes[k], elong[k] = (c0, v0), 0.0
            else:
                axes[k], elong[k] = r[:2], r[2]
    down = np.array([0.0, 1.0])
    for k, nb in (("4", "5"), ("5", "4")):
        if k in axes and nb in axes and elong[k] < ELONG_MIN <= elong[nb]:
            notes[f"{k}_unreliable"] = round(elong[k], 2)
            axes[k] = (axes[k][0], axes[nb][1])
    if "4" in axes and "5" in axes and angle(axes["4"][1], axes["5"][1]) > MAX_DIFF_V2:
        bad, good = ("4", "5") if elong["4"] < elong["5"] else ("5", "4")
        notes[f"{bad}_diff"] = round(angle(axes["4"][1], axes["5"][1]), 1)
        axes[bad] = (axes[bad][0], axes[good][1])
    if "6" in pts:
        ref = [axes[k][1] for k in "45" if k in axes]
        r = np.mean(ref, axis=0) if ref else down.copy()
        r = r / np.linalg.norm(r)
        n = distal_normal(r, quadrant)
        p6 = pts["6"]
        c0 = p6.mean(axis=0)
        rel = p6 - c0
        roots = []
        for side in (rel @ n < 0, rel @ n >= 0):  # 近心、遠心
            half = p6[side]
            ra = root_axis(half, half.mean(axis=0), r) if len(half) >= MIN_PX else None
            if ra is not None and ra[2] >= ELONG_MIN and angle(ra[1], r) <= MAX_DIFF_V2:
                roots.append(ra)
        lower = p6[rel @ r > 0]
        center = lower.mean(axis=0) if len(lower) >= MIN_PX else c0
        if roots:
            v = np.mean([ra[1] if ra[1][1] >= 0 else -ra[1] for ra in roots], axis=0)
            v = v / np.linalg.norm(v)
            if len(roots) == 1:
                notes["6_one_root"] = 1
        else:
            v = r
            notes["6_no_root"] = 1
        if "5" in axes and angle(v, axes["5"][1]) > MAX_DIFF_V2:
            notes["6_diff"] = round(angle(v, axes["5"][1]), 1)
            v = axes["5"][1]
        axes["6"] = (center, v)
    return {k: (c, v, distal_normal(v, quadrant)) for k, (c, v) in sorted(axes.items())}


def item_teeth(item):
    # 輪廓 → 遮罩像素(畫在加了 PAD 的畫布上,裁切影像外的牙根不會被截掉);回傳 {"4"/"5"/"6": 像素座標(裁切影像座標)}
    pts = {}
    for t in item["teeth"]:
        if len(t["fdi"]) != 2 or t["fdi"][1] not in "456":
            continue
        poly = np.asarray(t["poly"], dtype=float) + PAD
        x1, y1 = np.floor(poly.min(axis=0)).astype(int)
        x2, y2 = np.ceil(poly.max(axis=0)).astype(int) + 1
        mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
        cv2.fillPoly(mask, [np.round(poly - (x1, y1)).astype(np.int32)], 1)
        ys, xs = np.nonzero(mask)
        pts[t["fdi"][1]] = np.column_stack([xs, ys]).astype(float) + (x1, y1) - PAD
    return pts


def lines_v1(pts, quadrant, notes):
    # 現行版本: 直接呼叫 mf_position.reference_lines。tooth_pixels 讀 mask[bbox] 再加回 bbox 左上角,
    #   所以把像素平移到正座標、放進整張畫布,算完再平移回來
    shift = -np.floor(min(p.min(axis=0).min() for p in pts.values())) + 1
    lower = []
    for k, p in pts.items():
        q = (p + shift).astype(int)
        mask = np.zeros((q[:, 1].max() + 1, q[:, 0].max() + 1), np.uint8)
        mask[q[:, 1], q[:, 0]] = 1
        lower.append({"fdi_id": f"{quadrant}{k}", "bbox": (0, 0, mask.shape[1], mask.shape[0]), "mask": mask})
    return {k: (c - shift, v, n) for k, (c, v, n) in reference_lines(lower, quadrant, notes).items()}


def tilt(v):
    return angle(v, np.array([0.0, 1.0]))


def draw(alias, item, l1, l2, path):
    img = cv2.imread(str(IMAGES / item["image"]))
    h = img.shape[0]
    for t in item["teeth"]:
        cv2.polylines(img, [np.round(np.asarray(t["poly"])).astype(np.int32)], True, (90, 200, 90), 1)
    cv2.polylines(img, [np.round(np.asarray(item["mf"])).astype(np.int32)], True, (0, 200, 255), 2)
    for lines, color, dx in ((l1, (60, 60, 255), 0), (l2, (255, 220, 0), 0)):
        for k, (c, v, _) in lines.items():
            t1, t2 = (0 - c[1]) / v[1], (h - c[1]) / v[1]
            a, b = c + t1 * v, c + t2 * v
            cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), color, 1, cv2.LINE_AA)
            cv2.putText(img, k, tuple(np.round(c + (h * 0.35 - c[1]) / v[1] * v + (4, 0)).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(img, f"{alias}  red = v1 (current)  cyan = v2", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlays", type=int, default=30, help="判定改變的頦孔中隨機抽幾張畫疊圖(另加使用者的例子)")
    parser.add_argument("--also", nargs="*", default=["p004L", "p004R", "p003L", "p003R"])
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    data = json.loads(ITEMS_JSON.read_text())
    with open(KEY_CSV, newline="") as f:
        key = {r["alias"]: r for r in csv.DictReader(f)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "overlays").mkdir(exist_ok=True)

    rows, reproduce = [], Counter()
    for alias in data["order"]:
        item, k = data["items"][alias], key[alias]
        q = int(item["quadrant"])
        pts = item_teeth(item)
        if not all(x in pts for x in "456"):
            continue
        n1, n2 = {}, {}
        l1 = lines_v1(pts, q, n1)
        l2 = reference_lines_v2(pts, q, n2)
        mf = np.asarray(item["mf"], dtype=float)
        p1, _ = classify(mf, l1, "center")
        p2, _ = classify(mf, l2, "center")
        reproduce[str(p1) == k["auto_position_center"]] += 1
        rows.append({"alias": alias, "set": k["set"], "key_center": k["auto_position_center"], "v1": p1, "v2": p2,
                     "v1_fallback": " ".join(f"{a[4]}:{b}" for a, b in n1.items()), "v2_notes": " ".join(f"{a}:{b}" for a, b in n2.items()),
                     **{f"tilt{x}_v1": round(tilt(l1[x][1]), 1) for x in "456"}, **{f"tilt{x}_v2": round(tilt(l2[x][1]), 1) for x in "456"},
                     "_l": (l1, l2)})

    with open(OUT / "compare.csv", "w", newline="") as f:
        fields = [c for c in rows[0] if c != "_l"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    lines = [f"頦孔 {n} 個(x4–x6 都有);用重建的遮罩重算現行版,與 key.csv 判定相同 {reproduce[True]} / {n}"]
    for x in "456":
        fb1 = sum(f"{x}:" in r["v1_fallback"] for r in rows)
        fb2 = sum(any(t.startswith(f"{x}_") for t in r["v2_notes"].split()) for r in rows)
        t1 = [r[f"tilt{x}_v1"] for r in rows]
        t2 = [r[f"tilt{x}_v2"] for r in rows]
        lines.append(f"x{x}: 借用其他牙方向 v1 {fb1} ({fb1 / n:.0%}) → v2 {fb2} ({fb2 / n:.0%});與垂直夾角中位數 {np.median(t1):.1f}° → {np.median(t2):.1f}°,最斜 5% {np.percentile(t1, 95):.1f}° → {np.percentile(t2, 95):.1f}°")
    notes = Counter(t.split(":")[0] for r in rows for t in r["v2_notes"].split())
    lines.append(f"v2 的保險觸發: {dict(notes)}")
    changed = [r for r in rows if r["v1"] != r["v2"]]
    lines.append(f"中央一半判定改變 {len(changed)} / {n}(測試集 {sum(r['set'] == 'test' for r in changed)} / {sum(r['set'] == 'test' for r in rows)})")
    lines.append("改變方向: " + ", ".join(f"{a}→{b}: {c}" for (a, b), c in Counter((r["v1"], r["v2"]) for r in changed).most_common()))
    lines.append("分布 v1: " + str(dict(sorted(Counter(r["v1"] for r in rows).items()))) + "  v2: " + str(dict(sorted(Counter(r["v2"] for r in rows).items()))))
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    pick = random.Random(0).sample(changed, min(args.overlays, len(changed)))
    pick += [r for r in rows if r["alias"] in args.also and r not in pick]
    with open(OUT / "overlays.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "alias", "v1", "v2", "v1_fallback", "v2_notes", "better(v1/v2/same/neither)"])
        for i, r in enumerate(pick, 1):
            name = f"{i:02d}_{r['alias']}.jpg"
            draw(r["alias"], data["items"][r["alias"]], *r["_l"], OUT / "overlays" / name)
            w.writerow([name, r["alias"], r["v1"], r["v2"], r["v1_fallback"], r["v2_notes"], ""])
    print(f"疊圖 {len(pick)} 張 → {OUT / 'overlays'}")


if __name__ == "__main__":
    main()
