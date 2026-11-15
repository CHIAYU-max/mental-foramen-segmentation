"""2026-11-15 從已經算好的 per_side.csv / per_roi.csv 重做 deploy_pipeline.py --score 的統計與配對比較(伺服器 CPU)。
用途: GPU 版評分在筆電 RTX 4050 上算(run_gpu_score.ps1),標註留在伺服器,統計在這裡做。
用法: python -m deployment.score_from_csv score_gpu_fp32 [--baseline score_official]
輸出: compare_results/deploy_eval/<name>/summary.md、summary_mf.csv(、vs_baseline_mf.csv)
"""
import argparse
import csv

import deployment.deploy_pipeline as dp

INT = ("seed", "quadrant", "models_run", "mf_found", "flag_fdi", "flag_side", "flag_flip", "flag_margin", "refer")
FLOAT = ("mf_dice", "mf_hit_1R", "ian_dice")


def load(path, ints, floats):
    rows = []
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        for k in ints:
            if k in r and r[k] != "":
                r[k] = int(float(r[k]))
        for k in floats:
            if k in r and r[k] != "":
                r[k] = float(r[k])
        if r.get("margin", "") != "":
            r["margin"] = float(r["margin"])
        rows.append(r)
    return rows


parser = argparse.ArgumentParser()
parser.add_argument("name")
parser.add_argument("--baseline", default="score_official")
args = parser.parse_args()
out = dp.OUT / args.name
rows = load(out / "per_side.csv", INT, ())
mf_rows = load(out / "per_roi.csv", ("seed",), FLOAT)
(out / "summary.md").unlink(missing_ok=True)
dp.summarize_score(rows, mf_rows, out)
if args.baseline:
    dp.compare_baseline(rows, mf_rows, dp.OUT / args.baseline, out)
