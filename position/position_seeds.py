# 2026-11-05 實驗 2: 牙位正確率改用多種子,並補上硬體版本(只讀已存遮罩,輸出在 compare_results/mf_position/position_seeds/)
# 規則與 position_report 相同: v2 參考線(items.json lines_a)+ 中央一半;預測 = 測試集五折多數決 → lcc(predicted_mf)。
# 正確率 strict: 標註 1–6 的正常牙,沒給牙位算錯;不在新切分的 ROI 略過(同 mf_position_eval)。
# 配對: 每個頦孔先對種子取平均,差 = 變體 − R34/512 FP32,以全景片為單位 bootstrap 2000 次。
# R18/384 單一模型(全資料訓練)沒有存遮罩: 先用 distill/weights 推論到 distill/masks/test/<tag>/fold0/(GPU,已存在就略過)。
import csv, json, os, sys
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

ROOT = Path.home() / "YOLOv8"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from position.mf_position import ITEMS_JSON, KEY_CSV, OUT, centroid, classify
import position.mf_position_eval as ev

MAIN = ROOT / ev.MASKS
DST = ROOT / "compare_results" / "mf_position" / "position_seeds"
R18 = MAIN / "distill"
Q = ROOT / "compare_results" / "int8_pano"
sfx = lambda s: "" if s == 0 else f"_s{s}"
VARIANTS = [  # (名稱, 遮罩根目錄, 各種子的 tag)
    ("R34/512 FP32 五折(主結果)", MAIN, [f"r34_imagenet_aug{sfx(s)}" for s in range(5)]),
    ("OpenVINO FP32 對照", Q, [f"ov_fp32{sfx(s)}" for s in range(5)]),
    ("OpenVINO INT8 PTQ(CPU 部署)", Q, [f"ov_int8{sfx(s)}" for s in range(5)]),
    ("torch.ao PTQ", Q, [f"tq_ptq{sfx(s)}" for s in range(5)]),
    ("QAT", Q, [f"tq_qat{sfx(s)}" for s in range(5)]),
    ("QAT + 定點(RTL 路線)", Q, [f"tq_qat_fx{sfx(s)}" for s in range(5)]),
    ("slim+half 五折", MAIN, [f"r34_imagenet_aug_slimhalf{sfx(s)}" for s in range(5)]),
    ("R34/384 五折", MAIN, [f"r34_imagenet_aug_384{sfx(s)}" for s in range(5)]),
    ("R18/384 五折(3 種子)", MAIN, [f"r18_imagenet_aug_384{sfx(s)}" for s in range(3)]),
    ("R18/384 單一模型", R18, [f"r18_imagenet_aug_384_singlelcc_s{s}" for s in range(5)]),
    ("YOLO26s 五折", MAIN, [f"yolo26s-seg{sfx(s)}" for s in range(5)]),
]


def infer_r18():
    import torch
    from stage2.compare_yolo_unet import IMAGES_DIR, index_files
    from stage2.kfold_pano import EXCLUDE_TEST
    from stage2.kfold_unet_variants import CONFIGS, build_model, predict_masks
    from stage2.predict_unet import DEVICE
    cfg = CONFIGS["r18_imagenet_aug_384"]
    files = {s: p for s, p in index_files((ROOT / IMAGES_DIR).resolve()).items() if s not in EXCLUDE_TEST}
    for s in range(5):
        out = R18 / "masks" / "test" / f"r18_imagenet_aug_384_singlelcc_s{s}" / "fold0"
        if out.exists() and len(list(out.glob("*.png"))) == len(files):
            continue
        out.mkdir(parents=True, exist_ok=True)
        model = build_model({**cfg, "pretrained": False}).to(DEVICE)
        w = MAIN / "distill" / "weights" / f"r18_imagenet_aug_384_singlelcc_s{s}.pth"
        model.load_state_dict(torch.load(w, map_location=DEVICE)["model_state_dict"])
        model.eval()
        with torch.inference_mode():
            for stem, path in files.items():
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                pred = predict_masks(model, img, cfg["size"])
                cv2.imwrite(str(out / f"{stem}.png"), (pred["mf"].astype(np.uint8) | (pred["ian"].astype(np.uint8) << 1)))
        print("r18 single seed", s, "->", out, flush=True)
        del model


def margin(poly, lines):
    # 預測頦孔質心到最近判定邊界(|d| = 0.5 × 半寬)的距離,以頦孔半寬為單位
    m = centroid(poly)
    vals = []
    for c, _, n in lines.values():
        half = float(np.ptp((poly - m) @ n)) / 2
        d = float((m - c) @ n)
        vals.append(abs(abs(d) - 0.5 * half) / max(half, 1e-6))
    return min(vals)


def boot(per_pano, n=2000, seed=0):
    # per_pano: [(sum_a, count)]; 回傳 95% CI 與雙尾 p(對 0)
    rng = np.random.default_rng(seed)
    arr = np.array(per_pano, float)
    idx = rng.integers(0, len(arr), (n, len(arr)))
    s = arr[idx].sum(1)
    stat = s[:, 0] / s[:, 1]
    lo, hi = np.percentile(stat, [2.5, 97.5])
    p = min(1.0, 2 * min((stat <= 0).mean(), (stat >= 0).mean()))
    return lo, hi, p


def main():
    infer_r18()
    DST.mkdir(parents=True, exist_ok=True)
    items = json.loads(ITEMS_JSON.read_text(encoding="utf-8"))["items"]
    keys = [k for k in csv.DictReader(open(KEY_CSV, newline="")) if k["set"] == "test"]
    sources = {r["stem"]: r for r in csv.DictReader(open(ev.ROI_SOURCES, newline=""))}
    orig = {r["alias"]: r for r in csv.DictReader(open(OUT / "labels_rater.csv", newline="", encoding="utf-8-sig"))}
    adjd = ROOT / "compare_results/mf_position/labels_adjudicated_dir/labels_rater.csv"
    adj = {r["alias"]: r for r in csv.DictReader(open(adjd, newline="", encoding="utf-8-sig"))}
    rows = []
    for name, root, tags in VARIANTS:
        for s, tag in enumerate(tags):
            for k in keys:
                item = items[k["alias"]]
                cx, cy = int(k["crop_x"]), int(k["crop_y"])
                lines = ev.lines_from_item(item, cx, cy, "a")
                found, poly = ev.predicted_mf(root, tag, k["stem"], "test", None)
                if not found:
                    continue
                pos, mg = "", ""
                if poly is not None:
                    poly = poly + (int(sources[k["stem"]]["x"]), int(sources[k["stem"]]["y"]))
                    p, _ = classify(poly, lines, "center")
                    pos = p or ""
                    if p:
                        mg = round(margin(poly, lines), 4)
                rows.append({"variant": name, "seed": s, "alias": k["alias"], "pano": k["pano"], "pred": pos, "margin": mg,
                             "orig": orig.get(k["alias"], {}).get("position", ""), "adj": adj.get(k["alias"], {}).get("position", "")})
        print("done", name, flush=True)
    with open(DST / "per_seed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lines_out = []
    table = []  # summary.csv(build_results_xlsx.py 的「牙位」工作表讀這份)
    for lab in ("orig", "adj"):
        lines_out.append(f"\n=== 對{'原始' if lab == 'orig' else '複查後'}標註(正常牙 1–6,strict) ===")
        lines_out.append("| 版本 | 種子 | n | 正確率 平均 ± SD | 各種子 | 差(− 主結果)[95% CI] | p |")
        lines_out.append("| --- | --- | --- | --- | --- | --- | --- |")
        main_acc = None
        for name, _, tags in VARIANTS:
            sub = [r for r in rows if r["variant"] == name and r[lab] in tuple("123456")]
            per_seed = [np.mean([str(r["pred"]) == r[lab] for r in sub if r["seed"] == s]) for s in range(len(tags))]
            acc = defaultdict(list)
            for r in sub:
                acc[r["alias"]].append(str(r["pred"]) == r[lab])
            acc = {a: np.mean(v) for a, v in acc.items()}
            n = len(acc)
            diff = ""
            rec = {"labels": lab, "variant": name, "seeds": len(tags), "n": n, "mean": round(float(np.mean(per_seed)), 5),
                   "sd": round(float(np.std(per_seed, ddof=1)), 5), "per_seed": " / ".join(f"{x:.4f}" for x in per_seed),
                   "diff": "", "ci_lo": "", "ci_hi": "", "p": ""}
            if main_acc is None:
                main_acc, pano_of = acc, {r["alias"]: r["pano"] for r in sub}
            else:
                common = sorted(set(acc) & set(main_acc))
                pp = defaultdict(lambda: [0.0, 0])
                for a in common:
                    pp[pano_of[a]][0] += acc[a] - main_acc[a]
                    pp[pano_of[a]][1] += 1
                d = np.mean([acc[a] - main_acc[a] for a in common])
                lo, hi, p = boot(list(pp.values()))
                diff = f"{100 * d:+.2f} 點 [{100 * lo:+.2f}, {100 * hi:+.2f}] | {p:.2f}"
                rec.update(diff=round(float(d), 5), ci_lo=round(float(lo), 5), ci_hi=round(float(hi), 5), p=round(float(p), 4))
            table.append(rec)
            lines_out.append(f"| {name} | {len(tags)} | {n} | {100 * np.mean(per_seed):.1f} ± {100 * np.std(per_seed, ddof=1):.1f}% | "
                             + " / ".join(f"{100 * x:.1f}" for x in per_seed) + f" | {diff or '— | —'} |")
    text = "\n".join(lines_out)
    print(text)
    (DST / "summary.md").write_text(text + "\n", encoding="utf-8")
    with open(DST / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)



if __name__ == "__main__":
    main()
