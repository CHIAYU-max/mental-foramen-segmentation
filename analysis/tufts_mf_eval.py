"""2026-11-14 外部驗證: 用使用者標的 Tufts 頦孔(50 個 ROI)算正式部署版本在外部資料上的命中率(使用者要求;CPU;內部實驗,Tufts 授權未確認)

標註: 筆電 C:\\dental_sim\\tufts_mf_label\\ 用 LabelMe 標(多邊形 mf,看得到神經管再標 ian;看不出頦孔就勾 not_visible、不畫),
  存成同名 .json;`tufts_mf_label.cmd` 關掉 LabelMe 後會把 .json 傳到伺服器 external/tufts/label_mf50/ 並執行這支程式。
  沒有 .json = 還沒標;有 .json 但沒有 mf = 標註者看不出頦孔。
樣本(10/29 定的): compare_results/tufts/rois.csv 的恆牙 ROI 依模型分歧度 u 分五等分,每等分隨機 10 個(每張全景片最多一側)。
  → 母體估計 = 五個等分的命中率取平均(等分大小相同);CI 以等分內重抽 2000 次。
模型: 正式部署的第二階段(R34/512 INT8 串接式聯集 + lcc,種子 0–4),直接吃標註用的同一張 ROI PNG;另報加 3×3 中值濾波的版本。
事先定的讀法: 只報告,不調任何東西。對照 = 我們測試集同一設定的命中 0.920。n = 50,CI 很寬,只能看出「差很多」或「差不多」。
輸出: compare_results/tufts/mf_eval/(summary.md、per_roi.csv)。用法: python -m analysis.tufts_mf_eval(--status 只看標了幾個)
"""
import argparse
import csv
import json
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch

from stage2.compare_yolo_unet import ROOT, load_gt
from analysis.paper_metrics import image_metrics

DIR = ROOT / "external" / "tufts" / "label_mf50"
OUT = ROOT / "compare_results" / "tufts" / "mf_eval"
OURS = 0.920


def items():
    out = []
    for r in csv.DictReader(open(DIR / "list.csv", newline="", encoding="utf-8-sig")):
        stem = f"{r['no']}_{r['stem']}"
        js, status = DIR / f"{stem}.json", "unlabeled"
        if js.exists():
            data = json.loads(js.read_text(encoding="utf-8"))
            status = "mf" if any(s["label"] == "mf" for s in data["shapes"]) else "not_visible"
        out.append({**r, "file": stem, "status": status})
    return out


def main():
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()
    its = items()
    n = defaultdict(int)
    for it in its:
        n[it["status"]] += 1
    print(f"Tufts 頦孔標註: 共 {len(its)} 個;已標有頦孔 {n['mf']}、標註者看不出頦孔 {n['not_visible']}、還沒標 {n['unlabeled']}")
    if args.status or not n["mf"]:
        return
    import deployment.deploy_pipeline as D
    D.CASCADE_OUT = "union"
    labeled = [it for it in its if it["status"] != "unlabeled"]
    for it in labeled:
        it["img"] = cv2.imread(str(DIR / f"{it['file']}.png"), cv2.IMREAD_GRAYSCALE)
        it["gt"] = load_gt(it["file"], it["img"].shape, DIR, {})
    per = defaultdict(list)
    with torch.inference_mode():
        for seed in range(5):
            models = D.load_models(seed, args.threads)
            for it in labeled:
                for name, x in (("official", it["img"]), ("median3", cv2.medianBlur(it["img"], 3))):
                    pred, k = D.cascade(models, x, D.Clock())
                    m = image_metrics(pred, it["gt"]) if it["status"] == "mf" else {}
                    per[(name, it["file"])].append((m.get("mf_hit_1R", float("nan")), m.get("mf_dice", float("nan")), float(pred["mf"].any()), k))
            print(f"種子 {seed} 完成", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, lines = [], ["# Tufts 外部驗證: 頦孔命中(tufts_mf_eval.py)", "",
                       f"- 標註 {len(labeled)} / {len(its)} 個: 有頦孔 {n['mf']}、標註者看不出頦孔 {n['not_visible']}(不算命中率,另外列)、還沒標 {n['unlabeled']}。",
                       f"- 模型: 正式部署的第二階段(R34/512 INT8 串接式聯集 + lcc),5 種子平均。對照: 我們的測試集同一設定命中 {OURS:.3f}。", ""]
    rng = np.random.default_rng(0)
    for name in ("official", "median3"):
        mean = {it["file"]: np.nanmean(np.array(per[(name, it["file"])], float), axis=0) for it in labeled}
        mf = [it for it in labeled if it["status"] == "mf"]
        by_q = defaultdict(list)
        for it in mf:
            by_q[it["u_quintile"]].append(mean[it["file"]][0])
        strat = float(np.mean([np.mean(v) for v in by_q.values()]))
        boots = [np.mean([np.mean(rng.choice(v, len(v))) for v in by_q.values()]) for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        raw = float(np.mean([mean[it["file"]][0] for it in mf]))
        lines += [f"## {'正式部署' if name == 'official' else '正式部署 + 3×3 中值濾波'}", "",
                  f"- 有頦孔標註的 {len(mf)} 個: 命中 {raw:.3f}(未加權);**依分歧度五等分加權 {strat:.3f} [95% CI {lo:.3f}, {hi:.3f}]**;"
                  f"mf Dice {np.mean([mean[it['file']][1] for it in mf]):.3f};模型有輸出頦孔 {np.mean([mean[it['file']][2] for it in mf]):.3f};每個 ROI 跑 {np.mean([mean[it['file']][3] for it in mf]):.2f} 個模型",
                  "- 各等分(分歧度由低到高)的命中: " + "、".join(f"Q{q} {np.mean(v):.2f}(n = {len(v)})" for q, v in sorted(by_q.items())),
                  f"- 標註者看不出頦孔的 {n['not_visible']} 個: 模型仍輸出頦孔的比例 {np.mean([mean[it['file']][2] for it in labeled if it['status'] == 'not_visible']) if n['not_visible'] else float('nan'):.2f}",
                  f"- 與我們測試集的差: {100 * (strat - OURS):+.1f} 點(CI {100 * (lo - OURS):+.1f} 到 {100 * (hi - OURS):+.1f})", ""]
        for it in labeled:
            v = mean[it["file"]]
            rows.append({"variant": name, "no": it["no"], "stem": it["stem"], "u_quintile": it["u_quintile"], "status": it["status"],
                         "hit": v[0], "mf_dice": v[1], "mf_found": v[2], "models_run": v[3]})
    text = "\n".join(lines)
    (OUT / "summary.md").write_text(text + "\n", encoding="utf-8")
    with open(OUT / "per_roi.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(text)


if __name__ == "__main__":
    main()
