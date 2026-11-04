import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import position.mf_position_eval as ev
from position.mf_position import KEY_CSV, OUT, REPORT_DIR, kappas

# 牙位結果的總報告(2026-10-31): 讀 labels_<rater>.csv,跑 mf_position_eval 的各模型與各規則,寫成一份 report.txt。
#   主結果: 方法 A + 中央一半(使用者選定),U-Net R34;對照 YOLO26s、YOLOv8n。
#   正確率兩種分母: strict = 所有有標註的頦孔,預測不出牙位算錯(端到端);both = 只算模型有給牙位的(與 kappa 相同)。
#   CI: 以全景片為單位 bootstrap 2000 次(同一張的左右側不獨立)。
#   FDI 編號: 流程是先判牙位再按 A,axes_shown 會是 0,所以分母用全部判定的頦孔,假設每個都有開疊圖檢查。
#   規則變體只在有標註的資料上比較,不用來挑測試集結果;主規則維持事先選定的方法 A + 中央一半。
MODELS = ("r34_imagenet_aug", "yolo26s-seg", "yolov8n-seg")
METHODS = (("a", "center"), ("a", "contour"), ("b", "center"), ("b", "contour"))
TARGET = 132
SEED, BOOT = 0, 2000


def load_labels(folder):
    labels = {}
    for path in sorted(Path(folder).glob("labels_*.csv")):
        with open(path, newline="") as f:
            labels[path.stem[len("labels_"):]] = {r["alias"]: r for r in csv.DictReader(f)}
    return labels


def boot_ci(values, panos, n=BOOT):
    # values: {alias: 數值};以全景片重抽
    by_pano = defaultdict(list)
    for a, v in values.items():
        by_pano[panos[a]].append(v)
    groups = list(by_pano.values())
    rng = np.random.default_rng(SEED)
    stats = []
    for _ in range(n):
        picked = [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        stats.append(np.mean([v for g in picked for v in g]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def run_eval(folder, out):
    ev.OUT, ev.REPORT_DIR = Path(folder), Path(out)
    for model in MODELS:
        for method, rule in METHODS:
            sys.argv = ["mf_position_eval.py", "--model", model, "--method", method, "--rule", rule]
            ev.main()


def read_eval(out, model, method, rule):
    with open(Path(out) / f"eval_{model}_{method}_{rule}.csv", newline="") as f:
        return {r["alias"]: r for r in csv.DictReader(f)}


def section(lines, subset, rater, rows, keys, labels, out):
    panos = {a: k["pano"] for a, k in keys.items()}
    ids = [a for a, r in labels.items() if r["position"] in tuple("123456") and keys[a]["set"] == subset]
    skipped = Counter(r["position"] for a, r in labels.items() if keys[a]["set"] == subset and r["position"] not in tuple("123456"))
    if not ids:
        lines.append(f"\n## {subset}: 沒有 1–6 的標註")
        return
    lines.append(f"\n## {subset}(標註者 {rater}): 判定 1–6 的 {len(ids)} 個、{len({panos[a] for a in ids})} 張全景片;略過 {dict(skipped)}")
    lines.append("標註分布: " + str(dict(sorted(Counter(labels[a]["position"] for a in ids).items()))))

    results = {}
    for model in MODELS:
        data = read_eval(out, model, "a", "center")
        use = [a for a in ids if a in data]
        col = f"label_{rater}"
        correct = {a: int(str(data[a]["pred"]) == labels[a]["position"]) for a in use}
        rule = {a: int(str(data[a]["rule_gt"]) == labels[a]["position"]) for a in use}
        results[model] = correct
        lo, hi = boot_ci(correct, panos)
        rlo, rhi = boot_ci(rule, panos)
        n, acc, kappa, kappa_q = kappas([data[a][col] for a in use], [str(data[a]["pred"]) for a in use])
        causes = Counter(data[a].get(f"cause_{rater}", "") for a in use)
        lines.append(f"\n{model}: 共 {len(use)} 個")
        lines.append(f"  端到端(預測頦孔 + 規則)正確率 strict {np.mean(list(correct.values())):.3f}(95% CI {lo:.3f}–{hi:.3f})")
        lines.append(f"  只算有給牙位的 {n} 個: {acc:.3f}、kappa {kappa:.3f}、二次加權 {kappa_q:.3f}")
        lines.append(f"  規則套 GT 頦孔 {np.mean(list(rule.values())):.3f}(95% CI {rlo:.3f}–{rhi:.3f})")
        lines.append(f"  相鄰一類以內(strict) {np.mean([data[a]['pred'] in tuple('123456') and abs(int(data[a]['pred']) - int(labels[a]['position'])) <= 1 for a in use]):.3f}")
        lines.append(f"  錯誤來源: {dict(causes)}")
        wrong = Counter((labels[a]["position"], str(data[a]["pred"]) or "無") for a in use if not correct[a])
        lines.append("  錯誤方向(標註 → 預測): " + ", ".join(f"{x}→{y} ×{c}" for (x, y), c in wrong.most_common(8)))

    base, other = results[MODELS[0]], results[MODELS[1]]
    common = [a for a in base if a in other]
    diff = {a: base[a] - other[a] for a in common}
    lo, hi = boot_ci(diff, panos)
    lines.append(f"\nU-Net − YOLO26s(strict,配對): {np.mean(list(diff.values())):+.3f}(95% CI {lo:+.3f}–{hi:+.3f})")

    lines.append("\n規則變體(規則套 GT 頦孔 vs 標註;主規則事先選定為 a center,這裡只作參考):")
    for method, rule in METHODS:
        data = read_eval(out, MODELS[0], method, rule)
        use = [a for a in ids if a in data]
        lines.append(f"  方法 {method} + {rule}: {np.mean([str(data[a]['rule_gt']) == labels[a]['position'] for a in use]):.3f}(n = {len(use)})")

    shown = ids
    wrong = {a: int(str(labels[a].get("fdi_wrong")) == "1") for a in shown}
    if wrong:
        lo, hi = boot_ci({a: 1 - v for a, v in wrong.items()}, panos)
        lines.append(f"\nFDI 編號(假設每個都有開疊圖檢查): {len(shown)} 個中,標記錯誤 {sum(wrong.values())} 個,該側編號正確率 {1 - np.mean(list(wrong.values())):.3f}(95% CI {lo:.3f}–{hi:.3f})")
    else:
        lines.append("\nFDI 編號: 沒有開疊圖的標註,無法計算")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default=str(OUT), help="labels_*.csv 所在資料夾")
    parser.add_argument("--out", default=str(REPORT_DIR / "report_labels"))
    parser.add_argument("--count", action="store_true", help="只印測試集已判定的個數")
    args = parser.parse_args()
    with open(KEY_CSV, newline="") as f:
        keys = {r["alias"]: r for r in csv.DictReader(f)}
    raters = load_labels(args.labels)
    if args.count:
        print(max([sum(r["position"] != "" and keys[a]["set"] == "test" for a, r in v.items()) for v in raters.values()], default=0))
        return
    Path(args.out).mkdir(parents=True, exist_ok=True)
    report = Path(args.out) / "report.txt"
    if not raters:
        report.write_text("還沒有 labels_*.csv\n", encoding="utf-8")
        print(report.read_text(encoding="utf-8"))
        return
    run_eval(args.labels, args.out)
    lines = ["# 牙位結果報告", f"目標: 測試集 {TARGET} 個(工具顯示;其中 p027L = 209_Left_roi 不在新切分,不計)"]
    for rater, labels in raters.items():
        done = sum(r["position"] != "" and keys[a]["set"] == "test" for a, r in labels.items())
        lines.append(f"{rater}: 測試集完成 {done} / {TARGET}" + ("" if done >= TARGET else "(**未標完,結果只代表前面這些**)"))
        for subset in ("test", "train"):
            section(lines, subset, rater, None, keys, labels, args.out)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
