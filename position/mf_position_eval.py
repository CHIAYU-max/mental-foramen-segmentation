import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from stage2.kfold_pano import ROI_SOURCES
from position.mf_position import ITEMS_JSON, KEY_CSV, OUT, REPORT_DIR, centroid, classify, kappas
from analysis.paper_metrics import read_masks

# 牙位評估(人工 ROI,2026-10-25): 第二階段預測的頦孔 → 換回全景片座標 → mf_position.classify → 與牙位 GT 比較。
#   預測: kfold_pano 的遮罩。測試集取五折多數決;訓練集(CV)取該 ROI 所在那一折的模型。
#   多個連通區塊時取面積最大者(與 U-Net 輸出一個頦孔的假設一致)。
#   參考線直接用 mf_position prepare 存下的(items.json),與標註工具顯示的完全相同,不重跑牙齒模型。
# 比較(每個頦孔一列,輸出 compare_results/mf_position/eval_<tag>.csv):
#   rule_gt   規則套在 GT 頦孔上的牙位(= key.csv 的 auto_position)
#   pred      規則套在預測頦孔上的牙位
#   pred vs rule_gt: 只看頦孔預測誤差會不會改變牙位答案(不需要人工標註,現在就能算)
#   有 labels_<名字>.csv 時,再把 rule_gt、pred 與標註者比較,並把 pred 的錯誤分成:
#     mf_miss    預測沒有頦孔,或預測質心不在 GT 等效半徑內(與 mf_hit_1R 相同)
#     rule       規則套在 GT 頦孔上就已經錯(牙齒分割、FDI 或規則本身)
#     mf_shift   頦孔有命中、規則在 GT 上正確,但預測頦孔的偏差改變了牙位
MASKS = Path("compare_results") / "kfold_pano_mf_559"
FOLDS = 5


# (參考線方法, 判定規則) → key.csv 中規則套在 GT 頦孔上的欄位;方法 b = 垂直咬合平面(items.json 的 lines),a = 牙根主成分(lines_a)
GT_COLUMNS = {("a", "contour"): "auto_position", ("b", "contour"): "auto_position_perp",
              ("a", "center"): "auto_position_center", ("b", "center"): "auto_position_perp_center"}


def lines_from_item(item, crop_x, crop_y, method="b"):
    # 從 items.json 的線段(裁切座標)還原 classify 用的 (線上一點, 方向, 指向遠心的法向量)
    distal = -1.0 if item["quadrant"] == 4 else 1.0
    lines = {}
    for seg in item["lines" if method == "b" else "lines_a"]:
        p0, p1 = (np.array(p) + (crop_x, crop_y) for p in seg["p"])
        v = (p1 - p0) / np.linalg.norm(p1 - p0)
        n = np.array([v[1], -v[0]])
        if n[0] * distal < 0:
            n = -n
        lines[seg["name"][1]] = (p0, v, n)
    return lines


def predicted_mf(root, tag, stem, dataset, fold):
    # 回傳 (是否有預測檔, 最大區塊的輪廓或 None)
    if dataset == "test":
        preds = [read_masks(root, "test", tag, f, stem) for f in range(FOLDS) if (root / "masks" / "test" / tag / f"fold{f}" / f"{stem}.png").exists()]
        preds = [p["mf"] for p in preds if p is not None]
        mask = np.sum(preds, axis=0) > len(preds) / 2 if preds else None
    else:
        pred = read_masks(root, "cv", tag, fold, stem) if fold is not None else None
        mask = pred["mf"] if pred is not None else None
    if mask is None:
        return False, None
    if not mask.any():
        return True, None
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cs, _ = cv2.findContours((labels == largest).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return True, max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="r34_imagenet_aug", help="kfold_pano 的模型名稱(masks 底下的資料夾)")
    parser.add_argument("--method", choices=["a", "b"], default="a", help="參考線: a = 牙根主成分沿長軸延伸(預設,2026-10-27 使用者選定),b = 垂直咬合平面")
    parser.add_argument("--rule", choices=["contour", "center"], default="center", help="判定: contour = 碰到輪廓,center = 中央一半(預設)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    items = json.loads(ITEMS_JSON.read_text(encoding="utf-8"))["items"]
    with open(KEY_CSV, newline="") as f:
        keys = list(csv.DictReader(f))
    with open(ROI_SOURCES, newline="") as f:
        sources = {r["stem"]: r for r in csv.DictReader(f)}
    with open(root / MASKS / "folds.csv", newline="") as f:
        folds = {r["stem"]: int(r["fold"]) for r in csv.DictReader(f)}
    raters = {}
    for path in sorted(OUT.glob("labels_*.csv")):
        with open(path, newline="") as f:
            raters[path.stem[len("labels_"):]] = {r["alias"]: r["position"] for r in csv.DictReader(f) if r["position"]}

    rows = []
    for k in keys:
        item, src = items[k["alias"]], sources[k["stem"]]
        dataset = "test" if k["set"] == "test" else "cv"
        crop_x, crop_y = int(k["crop_x"]), int(k["crop_y"])
        gt_poly = np.array(item["mf"]) + (crop_x, crop_y)
        radius = (cv2.contourArea(gt_poly.astype(np.float32)) / np.pi) ** 0.5
        lines = lines_from_item(item, crop_x, crop_y, args.method)
        found, poly = predicted_mf(root / MASKS, args.model, k["stem"], dataset, folds.get(k["stem"]))
        if not found:
            # 不在 kfold_pano 的切分裡(例如測試集中與訓練資料同一張全景片而被移除的 ROI)
            continue
        row = {"alias": k["alias"], "set": k["set"], "stem": k["stem"], "rule_gt": k[GT_COLUMNS[(args.method, args.rule)]]}
        if poly is None:
            row.update(pred="", pred_reason="no_mf", center_px="", hit_1R=0)
        else:
            poly = poly + (int(src["x"]), int(src["y"]))
            position, info = classify(poly, lines, args.rule)
            error = float(np.linalg.norm(centroid(poly) - centroid(gt_poly)))
            row.update(pred=position or "", pred_reason=info.get("reason", ""), center_px=round(error, 1), hit_1R=int(error <= radius))
        for name, labels in raters.items():
            label = labels.get(k["alias"], "")
            row[f"label_{name}"] = label
            if label in "123456" and label:
                if not row["hit_1R"]:
                    cause = "mf_miss"
                elif str(row["rule_gt"]) != label:
                    cause = "rule"
                elif str(row["pred"]) != label:
                    cause = "mf_shift"
                else:
                    cause = "correct"
                row[f"cause_{name}"] = cause
        rows.append(row)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"eval_{args.model}_{args.method}_{args.rule}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(key for r in rows for key in r)))
        writer.writeheader()
        writer.writerows(rows)

    print(f"模型 {args.model}: 頦孔 {len(rows)} 個(另有 {len(keys) - len(rows)} 個不在 kfold_pano 的切分裡)→ {out}")
    for subset in ("test", "train"):
        sub = [r for r in rows if r["set"] == subset]
        name = "測試集(五折多數決)" if subset == "test" else "訓練集(CV)"
        both = [r for r in sub if str(r["rule_gt"]) and str(r["pred"])]
        same = sum(str(r["rule_gt"]) == str(r["pred"]) for r in both)
        print(f"\n{name}: {len(sub)} 個,預測命中 1R {sum(r['hit_1R'] for r in sub) / len(sub):.4f}")
        print(f"  預測頦孔與 GT 頦孔的牙位相同: {same} / {len(both)} = {same / max(len(both), 1):.4f}"
              f"(兩者都可判定;不同的 {len(both) - same} 個是頦孔預測誤差改變了牙位)")
        changed = Counter((str(r["rule_gt"]), str(r["pred"])) for r in both if str(r["rule_gt"]) != str(r["pred"]))
        if changed:
            print("  改變的方向(GT → 預測):", ", ".join(f"{a}→{b} ×{c}" for (a, b), c in changed.most_common()))
        for rater in raters:
            labeled = [r for r in sub if r.get(f"label_{rater}", "") in list("123456")]
            if not labeled:
                continue
            n, acc, kappa, kappa_q = kappas([r[f"label_{rater}"] for r in labeled], [str(r["pred"]) for r in labeled])
            _, acc_rule, _, _ = kappas([r[f"label_{rater}"] for r in labeled], [str(r["rule_gt"]) for r in labeled])
            print(f"  vs {rater}({len(labeled)} 個): 預測牙位正確率 {acc:.4f}(kappa {kappa:.3f}、二次加權 {kappa_q:.3f});"
                  f"規則套 GT 頦孔 {acc_rule:.4f}")
            print("    錯誤來源:", dict(Counter(r[f"cause_{rater}"] for r in labeled)))


if __name__ == "__main__":
    main()
