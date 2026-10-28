import csv
import statistics
from pathlib import Path

import cv2
import numpy as np

from compare_preprocessing import collect_images
from compare_yolo_unet import GT_DIR, IMAGES_DIR, NAMES, ROOT, index_files, load_gt
from kfold_patient import DATASET, PATIENT_MAP
from rerank_candidates import mask_distance
from show_failures import draw

# 檢查標註本身:轉檔錯誤、多畫或漏畫、頦孔沒有接在神經管上、頦孔不比周圍暗(可能畫錯位置)等,
# 並比較第一批、第二批與測試集的標註特徵,看兩批的標註方式差在哪裡。
OUT_DIR = ROOT / "compare_results" / "label_check"
# 頦孔與神經管的最短距離超過這個值(像素)視為沒有相連
MAX_GAP = 10
# 頦孔面積偏離同一組中位數的程度(以 MAD 為單位)超過這個值視為異常
AREA_Z = 3.5


def read_counts(txt):
    counts = {cls: 0 for cls in NAMES.values()}
    out_of_range = False
    for line in txt.read_text().splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        counts[NAMES[int(parts[0])]] += 1
        coords = np.array(parts[1:], dtype=np.float64)
        out_of_range |= bool((coords < -1e-3).any() or (coords > 1 + 1e-3).any())
    return counts, out_of_range


def darkness(image, mf):
    # 頦孔是透射區,應比周圍暗;回傳 (周圍平均 - 內部平均) / 周圍標準差,越大代表越明顯的暗區
    radius = max(2, int(round((mf.sum() / np.pi) ** 0.5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    ring = cv2.dilate(mf.astype(np.uint8), kernel).astype(bool) & ~mf
    inside, around = image[mf].astype(np.float64), image[ring].astype(np.float64)
    return float((around.mean() - inside.mean()) / (around.std() + 1e-6))


def core_darkness(image, mf, core=6, inner=12, outer=20):
    # 與標註邊界畫得寬或窄無關的暗度: 只用 GT 中心位置,比較中心半徑 core 像素的圓與 inner~outer 像素的環
    cy, cx = np.argwhere(mf).mean(0)
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    dist = np.hypot(yy - cy, xx - cx)
    inside, around = image[dist <= core].astype(np.float64), image[(dist >= inner) & (dist <= outer)].astype(np.float64)
    return float((around.mean() - inside.mean()) / (around.std() + 1e-6))


def main():
    with open(PATIENT_MAP, newline="") as f:
        batch_of = {r["stem"]: r["batch"] for r in csv.DictReader(f)}
    samples = [(p, p.parent.parent / "labels", batch_of[s]) for s, p in sorted(collect_images((ROOT / DATASET).resolve()).items())]
    test_gt = (ROOT / GT_DIR).resolve()
    samples += [(p, test_gt, "test") for p in index_files((ROOT / IMAGES_DIR).resolve()).values()]

    rows, images = [], {}
    for path, label_dir, batch in samples:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        gt = load_gt(path.stem, image.shape, label_dir, {})
        counts, out_of_range = read_counts(label_dir / f"{path.stem}.txt")
        row = {"filename": path.stem, "batch": batch, "mf_count": counts["mf"], "ian_count": counts["ian"], "out_of_range": out_of_range}
        row["mf_area"] = int(gt["mf"].sum())
        row["ian_area"] = int(gt["ian"].sum())
        both = gt["mf"].any() and gt["ian"].any()
        row["mf_ian_gap"] = round(mask_distance(gt["mf"], gt["ian"]), 1) if both else ""
        row["mf_darkness"] = round(darkness(image, gt["mf"]), 3) if gt["mf"].any() else ""
        row["mf_core_darkness"] = round(core_darkness(image, gt["mf"]), 3) if gt["mf"].any() else ""
        rows.append(row)
        images[path.stem] = (image, gt)

    # 頦孔面積在同一組內的離群程度(取對數後以中位數與 MAD 標準化)
    for batch in sorted({r["batch"] for r in rows}):
        group = [r for r in rows if r["batch"] == batch and r["mf_area"] > 0]
        logs = np.log([r["mf_area"] for r in group])
        median, mad = np.median(logs), np.median(np.abs(logs - np.median(logs))) * 1.4826
        for r, v in zip(group, logs):
            r["mf_area_z"] = round(float((v - median) / mad), 2)

    for r in rows:
        reasons = []
        if r["mf_count"] != 1 or r["ian_count"] != 1:
            reasons.append(f"多邊形數 mf={r['mf_count']} ian={r['ian_count']}")
        if r["out_of_range"]:
            reasons.append("座標超出影像範圍")
        if r["mf_ian_gap"] != "" and r["mf_ian_gap"] > MAX_GAP:
            reasons.append(f"頦孔未接神經管({r['mf_ian_gap']}px)")
        if r["mf_darkness"] != "" and r["mf_darkness"] < 0:
            reasons.append("頦孔不比周圍暗")
        if abs(r.get("mf_area_z", 0)) > AREA_Z:
            reasons.append(f"頦孔面積異常(z={r['mf_area_z']})")
        r["reasons"] = "; ".join(reasons)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["filename", "batch", "mf_count", "ian_count", "out_of_range", "mf_area", "mf_area_z", "ian_area", "mf_ian_gap", "mf_darkness", "mf_core_darkness", "reasons"]
    with open(OUT_DIR / "label_check.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print("各組標註的特徵(中位數;相連 = 頦孔與神經管距離不超過 %d 像素)" % MAX_GAP)
    print(f"{'組別':<8}{'n':>5}{'頦孔面積':>10}{'神經面積':>10}{'頦孔-神經距離':>14}{'相連比例':>10}{'頦孔暗度':>10}{'中心暗度':>10}{'被標記':>8}")
    for batch in ("first", "second", "test"):
        group = [r for r in rows if r["batch"] == batch]
        gaps = [r["mf_ian_gap"] for r in group if r["mf_ian_gap"] != ""]
        dark = [r["mf_darkness"] for r in group if r["mf_darkness"] != ""]
        core = [r["mf_core_darkness"] for r in group if r["mf_core_darkness"] != ""]
        print(
            f"{batch:<8}{len(group):>5}{statistics.median(r['mf_area'] for r in group):>12.0f}{statistics.median(r['ian_area'] for r in group):>12.0f}"
            f"{statistics.median(gaps):>14.1f}{sum(g <= MAX_GAP for g in gaps) / len(gaps):>12.1%}{statistics.median(dark):>12.2f}"
            f"{statistics.median(core):>12.2f}{sum(bool(r['reasons']) for r in group):>8}"
        )

    # 被標記的影像畫出 GT 輪廓,供人工確認
    flagged = [r for r in rows if r["reasons"]]
    flag_dir = OUT_DIR / "flagged"
    flag_dir.mkdir(parents=True, exist_ok=True)
    for r in flagged:
        image, gt = images[r["filename"]]
        empty = {cls: np.zeros_like(m) for cls, m in gt.items()}
        canvas = draw(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), gt, empty, f"{r['batch']} {r['filename']}")
        cv2.imwrite(str(flag_dir / f"{r['batch']}_{r['filename']}.jpg"), canvas)
    print(f"\n共標記 {len(flagged)} 張(原因寫在 label_check.csv 的 reasons 欄)")
    for reason in ("多邊形數", "座標超出", "未接神經管", "不比周圍暗", "面積異常"):
        by_batch = {b: sum(reason in r["reasons"] for r in flagged if r["batch"] == b) for b in ("first", "second", "test")}
        print(f"  {reason:<10} first {by_batch['first']:>3}  second {by_batch['second']:>3}  test {by_batch['test']:>3}")
    print(f"\n逐張結果: {OUT_DIR / 'label_check.csv'}\n被標記影像的 GT 疊圖: {flag_dir}(圖中只畫 GT,綠色為頦孔、青色為神經管)")


if __name__ == "__main__":
    main()
