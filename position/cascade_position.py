"""2026-11-04 串接式推論的輸出方式 × 牙位(CPU,只讀已存遮罩)。

cascade_output.py 比了 交集 / 聯集 / 第一個模型 的命中與 Dice,當時說「等牙位標註後再比」。這裡把每種輸出的頦孔套上
牙位規則(v2 參考線 + 中央一半),和複查後的牙位標註比較。
- 只用測試集「正常牙」(標註為 1–6)的頦孔;五個種子(r34_imagenet_aug、_s1–_s4)。
- 串接式與 cascade_output.py 相同: 先跑兩折(10 種組合都算)、兩者頦孔有無一致且 IoU ≥ 0.5 就停,停下時輸出
  交集 / 聯集 / 第一個模型;沒停就用五折多數決。頦孔一律只留最大連通區塊。
- 每個頦孔的正確率 = 所有 (種子 × 組合) 的平均;五折多數決是種子平均。配對比較用 seed_bootstrap.cluster_test(以病人為單位)。
- 輸出 compare_results/mf_position/cascade_position.csv 與 summary。屬探索性比較(測試集),不用來挑主結果。
"""
import csv
import itertools
import json
from multiprocessing import Pool

import cv2
import numpy as np

from deployment.cascade_infer import agree, vote
from stage2.compare_yolo_unet import ROOT
from stage2.kfold_pano import ROI_SOURCES
from stage2.kfold_yolo_unet import K_FOLDS
from position.mf_position import ITEMS_JSON, KEY_CSV, classify
from position.mf_position_eval import MASKS, lines_from_item
from analysis.paper_metrics import SIDE, read_masks
from analysis.seed_bootstrap import cluster_test

LABELS = ROOT / "compare_results" / "mf_position" / "labels_rater_adjudicated.csv"
OUT = ROOT / "compare_results" / "mf_position"
TAGS = ["r34_imagenet_aug"] + [f"r34_imagenet_aug_s{s}" for s in (1, 2, 3, 4)]
TAU = 0.5
MODES = ("full", "intersection", "union", "first")
ITEMS, CASES = {}, []


def position(mask, item, k, src):
    if mask is None or not mask.any():
        return None
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    keep = labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cs, _ = cv2.findContours(keep.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    poly = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float) + (int(src["x"]), int(src["y"]))
    lines = lines_from_item(item, int(k["crop_x"]), int(k["crop_y"]), "a")
    return str(classify(poly, lines, "center")[0])


def one(index):
    k, src, label = CASES[index]
    item = ITEMS[k["alias"]]
    acc = {m: [] for m in MODES}
    stops = []
    for tag in TAGS:
        preds = [read_masks(ROOT / MASKS, "test", tag, f, k["stem"]) for f in range(K_FOLDS)]
        full = position(vote(preds)["mf"], item, k, src) == label
        acc["full"].append(full)
        for i, j in itertools.permutations(range(K_FOLDS), 2):
            a, b = preds[i], preds[j]
            stop = agree([a, b], TAU)
            stops.append(stop)
            for mode in MODES[1:]:
                if not stop:
                    acc[mode].append(full)
                    continue
                mf = {"intersection": a["mf"] & b["mf"], "union": a["mf"] | b["mf"], "first": a["mf"]}[mode]
                acc[mode].append(position(mf, item, k, src) == label)
    return {"alias": k["alias"], "stem": k["stem"], "stop_rate": float(np.mean(stops)), **{m: float(np.mean(v)) for m, v in acc.items()}}


def main():
    ITEMS.update(json.loads(ITEMS_JSON.read_text(encoding="utf-8"))["items"])
    with open(KEY_CSV, newline="") as f:
        key = {r["alias"]: r for r in csv.DictReader(f)}
    with open(ROI_SOURCES, newline="") as f:
        sources = {r["stem"]: r for r in csv.DictReader(f)}
    with open(LABELS, newline="", encoding="utf-8-sig") as f:
        labels = {r["alias"]: r["position"] for r in csv.DictReader(f) if r["position"] in list("123456")}
    for alias, label in labels.items():
        k = key[alias]
        if k["set"] == "test" and (ROOT / MASKS / "masks" / "test" / TAGS[0] / "fold0" / f"{k['stem']}.png").exists():
            CASES.append((k, sources[k["stem"]], label))
    with Pool(16) as pool:
        rows = pool.map(one, range(len(CASES)))
    with open(OUT / "cascade_position.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = [f"測試集正常牙 {len(rows)} 個;5 種子;串接式先兩折(20 種順序),停下的比例 {np.mean([r['stop_rate'] for r in rows]):.3f}"]
    for m in MODES:
        lines.append(f"  {m:13s} 牙位正確 {np.mean([r[m] for r in rows]):.4f}")
    rng = np.random.default_rng(0)
    for m in MODES[1:]:
        per = {}
        for r in rows:
            g = SIDE.sub("", r["stem"])
            s, n = per.get(g, (0.0, 0))
            per[g] = (s + r[m] - r["full"], n + 1)
        est, lo, hi, p = cluster_test(list(per.values()), rng)
        lines.append(f"  {m} − 五折: {100 * est:+.2f} 點 [{100 * lo:+.2f}, {100 * hi:+.2f}] p {p:.3f}")
    (OUT / "cascade_position_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
