import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from stage2.compare_yolo_unet import ROOT
from stage1.fdi_cropped import assign_fdi_indices, process_and_separate_jaws
from stage2.kfold_pano import ROI_SOURCES
from deployment.pipeline_timing import PANO_DIR, TEETH_MODEL
from stage1.stage1_eval import MF_CLASS, MIN_GAP, MIN_SCORE, TEST_LABELS, TRAIN_LABELS, Sized

# 頦孔的牙位分類(2026-10-25):Tebo & Telford(1950)六分類,以牙齒長軸判定。
#   1 第一小臼齒(x4)前方          2 x4 下方(x4 長軸穿過頦孔)
#   3 x4 與 x5 之間                4 x5 下方(x5 長軸穿過頦孔)
#   5 x5 與 x6 之間                6 第一大臼齒(x6)下方
#   x = 象限 3(病人左側,影像右側)或 4(病人右側,影像左側)。
# 長軸: 牙齒實例遮罩像素的主成分(PCA),方向朝下;只用牙根(下半部),x6 用整顆牙(2026-10-28 起,細節見 reference_lines)。
# 判定(頦孔用多邊形輪廓,不只看中心點):
#   某條參考線穿過頦孔輪廓 → 該牙下方(2 / 4 / 6);同時穿過多條時取離頦孔質心最近的那條。
#   都沒穿過 → 依質心落在哪兩條參考線之間(1 / 3 / 5);在 x6 近心根遠心側也算 6。
#   x4、x5、x6 任一顆沒找到(缺牙或第一階段沒分到)→ 不判定。
# 子指令:
#   prepare  用 GT 頦孔與第一階段的牙齒,產生標註工具的影像與 mf_position/key.csv(含規則判定結果,標註者看不到)
#   report   比較標註者與規則、標註者之間的一致性(Cohen's kappa、二次加權 kappa)
# 標註工具: position_server.py(只用標準函式庫,可單獨複製到其他電腦執行)
OUT = ROOT / "mf_position"
IMAGES = OUT / "images"
KEY_CSV = OUT / "key.csv"
ITEMS_JSON = OUT / "items.json"
REPORT_DIR = ROOT / "compare_results" / "mf_position"
MAX_AXIS_DIFF = 15.0  # 度;與相鄰牙的長軸方向差超過此值就改用相鄰牙的方向
X6_AXIS = "whole"  # x6 的參考線: "whole" 整顆牙(2026-10-28 使用者決定)/ "mesial" 近心根(2026-10-25 版)
DEDUP_PX = 15  # 同一側兩個 GT 頦孔質心距離小於此值視為同一個(同一個 ROI 出現在不同批次)
CROP_DISTAL, CROP_MESIAL, CROP_UP, CROP_DOWN = 450, 100, 600, 250
FULL_WIDTH = 1220
SEED = 0


def tooth_pixels(tooth):
    x1, y1, x2, y2 = tooth["bbox"]
    ys, xs = np.nonzero(tooth["mask"][y1:y2, x1:x2])
    return np.column_stack([xs + x1, ys + y1]).astype(float)


def principal(pts):
    # 回傳 (中心, 朝下的單位方向向量)
    c = pts.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov((pts - c).T))
    v = vecs[:, -1]
    return c, (v if v[1] >= 0 else -v)


def distal_normal(v, quadrant):
    # 與方向 v 垂直、指向遠心的單位向量(象限 4 在影像左側,遠心 = 往左)
    n = np.array([v[1], -v[0]])
    return n if n[0] * (-1.0 if quadrant == 4 else 1.0) >= 0 else -n


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(abs(a @ b), -1, 1))))


def occlusal_perpendicular(lower, quadrant):
    # 方法 B: 用該側 x3–x7 牙冠頂端(外框上緣與質心 x)擬合局部咬合平面,回傳垂直於它、朝下的單位向量;不足兩顆時回傳垂直向下
    tops = [(t["cx"], t["bbox"][1]) for t in lower if t.get("fdi_id", "")[:1] == str(quadrant) and t["fdi_id"][1:] in "34567"]
    if len(tops) < 2:
        return np.array([0.0, 1.0])
    xs, ys = np.array(tops, dtype=float).T
    slope = np.polyfit(xs, ys, 1)[0]
    v = np.array([-slope, 1.0])
    return v / np.linalg.norm(v)


def reference_lines_perp(lower, quadrant):
    # 方法 B(2026-10-25,對照用): 參考線垂直於局部咬合平面,通過牙根中心(下半部像素的平均;x6 只用近心根)。
    #   不受單顆牙遮罩形狀影響;但牙齒本身真的傾斜時會忽略傾斜。等牙位 GT 決定用哪個方法。
    v = occlusal_perpendicular(lower, quadrant)
    n = distal_normal(v, quadrant)
    by_fdi = {t["fdi_id"]: t for t in lower if "fdi_id" in t}
    lines = {}
    for k in "456":
        if f"{quadrant}{k}" not in by_fdi:
            continue
        pts = tooth_pixels(by_fdi[f"{quadrant}{k}"])
        rel = pts - pts.mean(axis=0)
        root = pts[(rel @ v > 0) & ((rel @ n < 0) if k == "6" else True)]
        lines[k] = ((root if len(root) >= 20 else pts).mean(axis=0), v, n)
    return lines


def reference_lines(lower, quadrant, notes=None):
    # {"4"/"5"/"6": (線上一點, 方向, 指向遠心的法向量)};法向量決定有號距離的正負: 正 = 遠心側
    # 2026-10-25 改版(使用者發現 36 的參考線太斜;全部 614 個 x6 參考線與垂直夾角中位數 27°、最斜的 5% 超過 49°):
    #   小臼齒: 先用整顆牙的主成分定出上下,再只取下半部(牙根)重算主成分,避開牙冠形狀與咬合面的影響。
    #   x6: 大臼齒冠寬、雙根,整顆的主成分不穩定。改用 x4 / x5 的平均方向當「上下」,把遮罩切成近心 / 遠心兩半,
    #       只取近心那一半的下半部(= 近心根)算主成分,直接得到近心根的長軸(舊版是整顆長軸往近心平移 1/4 牙寬)。
    #   保險: 與相鄰牙(x4 ↔ x5、x6 → x5)的方向差超過 MAX_AXIS_DIFF 度時,改用相鄰牙的方向(中心不變),記在 notes。
    by_fdi = {t["fdi_id"]: t for t in lower if "fdi_id" in t}
    pts = {k: tooth_pixels(by_fdi[f"{quadrant}{k}"]) for k in "456" if f"{quadrant}{k}" in by_fdi}
    axes = {}
    for k in "45":
        if k not in pts:
            continue
        c0, v0 = principal(pts[k])
        root = pts[k][(pts[k] - c0) @ v0 > 0]
        axes[k] = principal(root) if len(root) >= 20 else (c0, v0)
    if "6" in pts:
        ref = [axes[k][1] for k in "45" if k in axes]
        r = np.mean(ref, axis=0) if ref else np.array([0.0, 1.0])
        r = r / np.linalg.norm(r)
        n = distal_normal(r, quadrant)
        c0 = pts["6"].mean(axis=0)
        rel = pts["6"] - c0
        # X6_AXIS = "whole"(2026-10-28 使用者決定,預設): 整顆牙的長軸 = 下半部(兩個牙根)的主成分,與小臼齒的做法相同,
        #   上下仍用 x4 / x5 的平均方向來切(整顆大臼齒的主成分不穩定);"mesial": 2026-10-25 的近心根版本
        root = pts["6"][(rel @ r > 0) & ((rel @ n < 0) if X6_AXIS == "mesial" else True)]
        axes["6"] = principal(root) if len(root) >= 20 else (c0, r)
    def fallback(k, nb):
        # 把 k 的方向換成 nb 的方向
        if notes is not None:
            notes[f"axis{k}_fallback"] = round(angle(axes[k][1], axes[nb][1]), 1)
        axes[k] = (axes[k][0], axes[nb][1])

    down = np.array([0.0, 1.0])
    if "4" in axes and "5" in axes and angle(axes["4"][1], axes["5"][1]) > MAX_AXIS_DIFF:
        # 兩顆小臼齒不一致時,相信比較接近垂直的那一顆
        bad, good = ("4", "5") if angle(axes["4"][1], down) > angle(axes["5"][1], down) else ("5", "4")
        fallback(bad, good)
    if "6" in axes and "5" in axes and angle(axes["6"][1], axes["5"][1]) > MAX_AXIS_DIFF:
        fallback("6", "5")
    return {k: (c, v, distal_normal(v, quadrant)) for k, (c, v) in sorted(axes.items())}


def centroid(poly):
    m = cv2.moments(np.asarray(poly, dtype=np.float32))
    return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]]) if m["m00"] else np.asarray(poly).mean(axis=0)


def classify(poly, lines, rule="contour", frac=0.5):
    # 回傳 (類別 1–6 或 None, {原因或各參考線到質心的有號距離 px})
    # rule = contour: 參考線碰到頦孔輪廓(含相切)就算「該牙下方」
    #        center : 2026-10-25 使用者選定「中央一半」: 質心到參考線的距離 d ≤ frac × r 才算「該牙下方」,
    #                 r = 頦孔在垂直參考線方向的半寬;只擦到邊緣或相切算「兩牙之間」。不受頦孔畫大畫小影響
    # 兩種規則都是: 同時符合多條時取離質心最近的那條;都不符合時依質心落在哪兩條參考線之間
    missing = [k for k in "456" if k not in lines]
    if missing:
        return None, {"reason": "missing_" + "".join(missing)}
    poly = np.asarray(poly, dtype=float)
    m = centroid(poly)
    dist = {k: float((m - c) @ n) for k, (c, _, n) in lines.items()}
    if rule == "center":
        half = {k: float(np.ptp((poly - m) @ n)) / 2 for k, (_, _, n) in lines.items()}
        crossing = [k for k in lines if abs(dist[k]) <= frac * half[k]]
    else:
        crossing = [k for k, (c, _, n) in lines.items() if ((poly - c) @ n).min() <= 0 <= ((poly - c) @ n).max()]
    if crossing:
        position = {"4": 2, "5": 4, "6": 6}[min(crossing, key=lambda k: abs(dist[k]))]
    elif dist["4"] < 0:
        position = 1
    elif dist["5"] < 0:
        position = 3
    elif dist["6"] < 0:
        position = 5
    else:
        position = 6
    return position, {f"d{k}": round(d, 1) for k, d in dist.items()}


def load_mf():
    # {來源全景片: [{stem, set, poly}]}(全景片座標);篩選條件與 stage1_eval.load_gt 相同
    mfs = defaultdict(list)
    with open(ROI_SOURCES, newline="") as f:
        for r in csv.DictReader(f):
            if not r["source_pano"] or float(r["score"]) < MIN_SCORE:
                continue
            if r["second_score"] and float(r["score"]) - float(r["second_score"]) < MIN_GAP:
                continue
            label = (TRAIN_LABELS if r["set"] == "train" else TEST_LABELS) / f"{r['stem']}.txt"
            if not label.exists():
                continue
            x, y, w, h = (int(r[k]) for k in "xywh")
            for line in label.read_text().splitlines():
                values = line.split()
                if values and int(values[0]) == MF_CLASS:
                    poly = np.array(values[1:], dtype=float).reshape(-1, 2) * (w, h) + (x, y)
                    mfs[r["source_pano"]].append({"stem": r["stem"], "set": r["set"], "poly": poly})
    return mfs


def contour(tooth):
    x1, y1, x2, y2 = tooth["bbox"]
    cs, _ = cv2.findContours(tooth["mask"][y1:y2, x1:x2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 1.5, True).reshape(-1, 2) + (x1, y1)
    return c


def prepare(args):
    mfs = load_mf()
    panos = sorted(mfs)
    random.Random(SEED).shuffle(panos)
    # 測試集的全景片排前面,先標完就能評估
    panos.sort(key=lambda p: not any(m["set"] == "test" for m in mfs[p]))
    panos = panos[:args.limit or None]
    IMAGES.mkdir(parents=True, exist_ok=True)
    yolo = Sized(YOLO(TEETH_MODEL), 1024)
    keys, items = [], {}
    for i, pano in enumerate(panos, 1):
        img = cv2.imread(str(PANO_DIR / pano))
        h, w = img.shape[:2]
        _, lower = process_and_separate_jaws(img, yolo)
        lower = assign_fdi_indices(lower) if lower else []
        q4 = [t["cx"] for t in lower if t.get("quadrant") == 4]
        q3 = [t["cx"] for t in lower if t.get("quadrant") == 3]
        mid = (max(q4) + min(q3)) / 2 if q4 and q3 else w / 2
        pano_alias = f"p{i:03d}"
        scale = FULL_WIDTH / w
        cv2.imwrite(str(IMAGES / f"{pano_alias}_full.jpg"), cv2.resize(img, (FULL_WIDTH, round(h * scale)), interpolation=cv2.INTER_AREA))
        seen = []
        # 同一側重複的 GT 只留一個,測試集優先
        for mf in sorted(mfs[pano], key=lambda m: m["set"] != "test"):
            m = centroid(mf["poly"])
            quadrant = 4 if m[0] <= mid else 3
            if any(q == quadrant and np.linalg.norm(m - c) < DEDUP_PX for q, c in seen):
                continue
            seen.append((quadrant, m))
            side = "R" if quadrant == 4 else "L"
            alias = f"{pano_alias}{side}"
            notes = {}
            lines = reference_lines(lower, quadrant, notes)
            position, info = classify(mf["poly"], lines)
            lines_perp = reference_lines_perp(lower, quadrant)
            position_perp, _ = classify(mf["poly"], lines_perp)
            position_center, _ = classify(mf["poly"], lines, "center")
            position_perp_center, _ = classify(mf["poly"], lines_perp, "center")
            if quadrant == 4:
                x1, x2 = m[0] - CROP_DISTAL, mid + CROP_MESIAL
            else:
                x1, x2 = mid - CROP_MESIAL, m[0] + CROP_DISTAL
            x1, x2 = int(max(0, x1)), int(min(w, x2))
            y1, y2 = int(max(0, m[1] - CROP_UP)), int(min(h, m[1] + CROP_DOWN))
            cv2.imwrite(str(IMAGES / f"{alias}.jpg"), img[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 92])
            off = np.array([x1, y1])
            def segments(ls):
                segs = []
                for k, (c, v, _) in ls.items():
                    t1, t2 = (y1 - c[1]) / v[1], (y2 - c[1]) / v[1]
                    segs.append({"name": f"{quadrant}{k}" + ("近心根" if k == "6" and X6_AXIS == "mesial" else ""),
                                 "p": np.round([c + t1 * v - off, c + t2 * v - off], 1).tolist()})
                return segs
            items[alias] = {
                "image": f"{alias}.jpg", "full": f"{pano_alias}_full.jpg", "quadrant": quadrant,
                "crop": [x1 * scale, y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale],
                "mf": np.round(mf["poly"] - off, 1).tolist(),
                "teeth": [{"fdi": t["fdi_id"], "poly": (contour(t) - off).tolist()} for t in lower if t.get("quadrant") == quadrant],
                # lines = 方法 B(垂直咬合平面,對照);lines_a = 方法 A(牙根主成分,標註工具顯示這組,2026-10-27 使用者選定)
                "lines": segments(lines_perp),
                "lines_a": segments(lines),
            }
            keys.append({"alias": alias, "pano": pano, "side": side, "quadrant": quadrant, "stem": mf["stem"], "set": mf["set"],
                         "auto_position": position or "", "auto_position_perp": position_perp or "",
                         "auto_position_center": position_center or "", "auto_position_perp_center": position_perp_center or "",
                         "reason": info.get("reason", ""),
                         **{f"d{k}": info.get(f"d{k}", "") for k in "456"},
                         "axis_fallback": " ".join(f"{k[4]}:{v}" for k, v in notes.items()),
                         "mf_x": round(m[0], 1), "mf_y": round(m[1], 1), "crop_x": x1, "crop_y": y1})
        if i % 20 == 0 or i == len(panos):
            print(f"[{i}/{len(panos)}] {pano}")
    with open(KEY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(keys[0]))
        writer.writeheader()
        writer.writerows(keys)
    ITEMS_JSON.write_text(json.dumps({"order": [k["alias"] for k in keys], "items": items}, ensure_ascii=False))
    print(f"全景片 {len(panos)} 張、頦孔 {len(keys)} 個(測試集 {sum(k['set'] == 'test' for k in keys)})")
    print("規則判定:", dict(sorted(Counter(str(k["auto_position"] or k["reason"]) for k in keys).items())))


def kappas(a, b):
    from sklearn.metrics import cohen_kappa_score
    pairs = [(x, y) for x, y in zip(a, b) if x in "123456" and y in "123456"]
    if len(pairs) < 2:
        return len(pairs), float("nan"), float("nan"), float("nan")
    x, y = (list(map(int, v)) for v in zip(*pairs))
    acc = sum(i == j for i, j in pairs) / len(pairs)
    return len(pairs), acc, cohen_kappa_score(x, y), cohen_kappa_score(x, y, weights="quadratic")


def report(args):
    with open(KEY_CSV, newline="") as f:
        keys = {r["alias"]: r for r in csv.DictReader(f)}
    raters = {}
    for path in sorted(OUT.glob("labels_*.csv")):
        with open(path, newline="") as f:
            raters[path.stem[len("labels_"):]] = {r["alias"]: r["position"] for r in csv.DictReader(f) if r["position"]}
    if not raters:
        raise SystemExit(f"{OUT} 底下沒有 labels_*.csv,請先用 position_server.py 標註")
    # rule = 方法 A(牙根 / 近心根的主成分),rule_perp = 方法 B(垂直於咬合平面)
    # _center = 「中央一半」規則(classify rule="center")
    sources = {name: {a: str(k.get(col, "")) for a, k in keys.items()} for name, col in (
        ("rule", "auto_position"), ("rule_perp", "auto_position_perp"),
        ("rule_center", "auto_position_center"), ("rule_perp_center", "auto_position_perp_center"))}
    sources.update(raters)
    rows = []
    names = list(sources)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for subset in ("all", "test"):
                aliases = [x for x in keys if x in sources[a] and x in sources[b] and (subset == "all" or keys[x]["set"] == "test")]
                n, acc, k, kw = kappas([sources[a][x] for x in aliases], [sources[b][x] for x in aliases])
                rows.append({"a": a, "b": b, "subset": subset, "n": n, "accuracy": round(acc, 4), "kappa": round(k, 4), "kappa_quadratic": round(kw, 4)})
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_DIR / "agreement.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(" | ".join(rows[0]))
    for r in rows:
        print(" | ".join(str(v) for v in r.values()))
    for name, labels in raters.items():
        print(f"\n{name}: 已標 {len(labels)} / {len(keys)},分布 {dict(sorted(Counter(labels.values()).items()))}")
        print("混淆矩陣(列 = 標註者,欄 = 規則):")
        cats = ["1", "2", "3", "4", "5", "6", ""]
        print("     " + " ".join(f"{c or '-':>3}" for c in cats))
        for c in "123456MX":
            counts = Counter(sources["rule"][x] for x, v in labels.items() if v == c)
            if counts:
                print(f"  {c:>2} " + " ".join(f"{counts.get(r, 0):>3}" for r in cats))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--limit", type=int, default=0, help="只處理前 N 張全景片(試跑用)")
    sub.add_parser("report")
    args = parser.parse_args()
    {"prepare": prepare, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
