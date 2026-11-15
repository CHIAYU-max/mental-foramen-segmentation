# 2026-11-05 中線規則的理論檢驗(不用人工牙位標註;輸出在 compare_results/mf_position/midline/)
# 用 32 類 FDI 模型(Tufts)的牙齒遮罩當獨立參考:
#   每顆牙 PCA 長軸;沿長軸把牙冠最寬處(上 40%)與根尖附近(下 20%)的近遠心寬度往下延伸成「影子」
#   頦孔質心落在幾顆牙的影子裡: 0 = 間隙、1 = 某顆牙下方、2 = 重疊
# 比較: 現行「中央一半」與中線規則,參考線分別用第一階段(items.json lines_a)與 32 類遮罩長軸
import csv, json, pickle, sys
from collections import Counter
from pathlib import Path
import cv2
import numpy as np

ROOT = Path.home() / "YOLOv8"
sys.path.insert(0, str(ROOT))
from position.mf_position import ITEMS_JSON, KEY_CSV, centroid, classify
import position.mf_position_eval as ev
from stage1.teeth32_compare import WEIGHTS, FDI
from deployment.pipeline_timing import PANO_DIR
from position.midline_rule import classify_mid

OUT = ROOT / "compare_results" / "mf_position" / "midline"
CACHE = OUT / "teeth32_masks.pkl"
POS = {"4": 2, "5": 4, "6": 6}
BETWEEN = {("4", "5"): 3, ("5", "6"): 5}


def masks32(model, img):
    r = model.predict(img, imgsz=1024, conf=0.25, verbose=False, retina_masks=True)[0]
    best = {}
    if r.masks is None:
        return {}
    for poly, cls, conf in zip(r.masks.xy, r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()):
        if len(poly) >= 10 and (cls not in best or conf > best[cls][1]):
            best[cls] = (np.asarray(poly, dtype=float), float(conf))
    return {FDI[c]: p for c, (p, _) in best.items()}


def tooth_geom(poly, distal):
    # 填滿遮罩取像素做 PCA;回傳 (中心, 長軸單位向量(往根尖=往下), 指向遠心的法向量, 牙冠半寬, 根尖半寬)
    x0, y0 = np.floor(poly.min(0)).astype(int)
    x1, y1 = np.ceil(poly.max(0)).astype(int)
    m = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    cv2.fillPoly(m, [(poly - (x0, y0)).astype(np.int32)], 1)
    ys, xs = np.nonzero(m)
    pts = np.stack([xs + x0, ys + y0], 1).astype(float)
    c = pts.mean(0)
    w, v = np.linalg.eigh(np.cov((pts - c).T))
    a = v[:, 1]
    if a[1] < 0:
        a = -a
    n = np.array([a[1], -a[0]])
    if n[0] * distal < 0:
        n = -n
    t = (pts - c) @ a
    s = (pts - c) @ n
    lo, hi = t.min(), t.max()
    crown = s[t <= lo + 0.4 * (hi - lo)]
    apex = s[t >= hi - 0.2 * (hi - lo)]
    return c, a, n, np.ptp(crown) / 2, np.ptp(apex) / 2, float(np.mean(crown)), float(np.mean(apex))


def shadow_class(m, geo, part):
    # part: "crown" / "apex";回傳 (類別或 'overlap', 落在哪些牙)
    inside = []
    for k in "3456":
        if k not in geo:
            continue
        c, a, n, wc, wa, oc, oa = geo[k]
        s = (m - c) @ n
        off, w = (oc, wc) if part == "crown" else (oa, wa)
        if abs(s - off) <= w:
            inside.append(k)
    if len(inside) >= 2:
        return "overlap", inside
    if len(inside) == 1:
        return {"3": 1, "4": 2, "5": 4, "6": 6}[inside[0]], inside
    # 間隙: 依質心在哪兩條長軸之間
    d = {k: float((m - geo[k][0]) @ geo[k][2]) for k in "456" if k in geo}
    if len(d) < 3:
        return None, inside
    if d["4"] < 0:
        return 1, inside
    if d["5"] < 0:
        return 3, inside
    if d["6"] < 0:
        return 5, inside
    return "beyond6", inside


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads(ITEMS_JSON.read_text(encoding="utf-8"))
    key = {r["alias"]: r for r in csv.DictReader(open(KEY_CSV, newline=""))}
    cache = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else {}
    model = None
    rows = []
    for alias in data["order"]:
        k, item = key[alias], data["items"][alias]
        if k["pano"] not in cache:
            if model is None:
                from ultralytics import YOLO
                model = YOLO(WEIGHTS)
            cache[k["pano"]] = masks32(model, cv2.imread(str(PANO_DIR / k["pano"])))
        q = str(k["quadrant"])
        distal = -1.0 if q == "4" else 1.0
        cx, cy = int(k["crop_x"]), int(k["crop_y"])
        mf = np.array(item["mf"], dtype=float) + (cx, cy)
        m = centroid(mf)
        teeth = cache[k["pano"]]
        geo = {n: tooth_geom(teeth[q + n], distal) for n in "3456" if q + n in teeth}
        row = {"alias": alias, "set": k["set"]}
        l1 = ev.lines_from_item(item, cx, cy, "a")
        row["cur_s1"] = classify(mf, l1, "center")[0]
        row["mid_s1"] = classify_mid(mf, l1, 0.5)
        if all(n in geo for n in "456"):
            l32 = {n: (geo[n][0], geo[n][1], geo[n][2]) for n in "456"}
            row["cur_32"] = classify(mf, l32, "center")[0]
            row["mid_32"] = classify_mid(mf, l32, 0.5)
            row["crown"], row["crown_in"] = shadow_class(m, geo, "crown")
            row["apex"], row["apex_in"] = shadow_class(m, geo, "apex")
            row["r"] = ev.cv2.contourArea(mf.astype(np.float32)) ** 0.5 / np.pi ** 0.5
            d4, d5 = ((m - geo[n][0]) @ geo[n][2] for n in "45")
            row["gap45_r"] = (d4 - d5) / row["r"]
            row["crown45_r"] = (geo["4"][3] + geo["5"][3]) / row["r"]  # 兩顆牙冠半寬和(= 牙冠緊貼時的軸距)
            row["apex45_r"] = (geo["4"][4] + geo["5"][4]) / row["r"]
        rows.append(row)
    CACHE.write_bytes(pickle.dumps(cache))
    ok = [r for r in rows if "crown" in r and r["cur_s1"] is not None]
    print(f"頦孔 {len(rows)} 個,32 類模型 x4–x6 都有且第一階段可判定: {len(ok)}")
    print("尺度(以頦孔半寬 r 為單位,中位數 [P10, P90]):")
    for f, name in (("gap45_r", "x4–x5 長軸在頦孔處的距離"), ("crown45_r", "34+35 牙冠半寬和"), ("apex45_r", "34+35 根尖半寬和")):
        v = [r[f] for r in ok]
        print(f"  {name}: {np.median(v):.2f} [{np.percentile(v, 10):.2f}, {np.percentile(v, 90):.2f}]")
    table, shadow = [], []  # summary_agree.csv、summary_shadow.csv(build_results_xlsx.py 讀)
    for part, name in (("crown", "牙冠寬度往下延伸"), ("apex", "根尖寬度往下延伸")):
        n_in = Counter(len(r[part + "_in"]) for r in ok)
        print(f"\n[{name}] 頦孔落在幾顆牙的影子裡: 0(間隙)={n_in[0]}  1={n_in[1]}  ≥2(重疊)={sum(v for c, v in n_in.items() if c >= 2)}  / {len(ok)}")
        print("  參考類別分布:", dict(sorted(Counter(str(r[part]) for r in ok).items())))
        single = [r for r in ok if r[part] != "overlap"]
        shadow.append({"reference": part, "one_tooth": n_in[1], "gap": n_in[0],
                       "overlap": sum(v for c, v in n_in.items() if c >= 2), "n": len(ok)})
        for rule in ("cur_s1", "mid_s1", "cur_32", "mid_32"):
            agree = sum(str(r[rule]) == str(r[part]) for r in single)
            table.append({"reference": part, "rule": rule, "agree": agree, "n": len(single)})
            print(f"  {rule:7s} 與參考一致 {agree}/{len(single)} = {agree / len(single):.3f}(排除重疊)")
    print("\n各規則分布:")
    for rule in ("cur_s1", "mid_s1", "cur_32", "mid_32"):
        print(f"  {rule:7s}", dict(sorted(Counter(str(r[rule]) for r in ok).items())))
    for rule in ("cur", "mid"):
        a = sum(str(r[rule + "_s1"]) == str(r[rule + "_32"]) for r in ok)
        print(f"  {rule}: 第一階段參考線 vs 32 類長軸 判定相同 {a}/{len(ok)} = {a / len(ok):.3f}")
        table.append({"reference": "stage1_vs_teeth32", "rule": rule, "agree": a, "n": len(ok)})
    for name, data in (("summary_agree.csv", table), ("summary_shadow.csv", shadow)):
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    with open(OUT / "midline_teeth32.csv", "w", newline="") as f:
        cols = ["alias", "set", "cur_s1", "mid_s1", "cur_32", "mid_32", "crown", "crown_in", "apex", "apex_in", "r", "gap45_r", "crown45_r", "apex45_r"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)



if __name__ == "__main__":
    main()
