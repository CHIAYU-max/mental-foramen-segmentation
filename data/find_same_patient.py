import csv
import re

import cv2
import numpy as np
import torch
import torchvision

from stage2.compare_yolo_unet import IMAGES_DIR, ROOT, index_files
from stage2.kfold_patient import PATIENT_MAP
from stage2.predict_unet import DEVICE

# 找出測試集與訓練資料中可能是同一位病人、不同時期拍攝的全景片(例如植牙前後)。
# 編號不同所以先前的檢查抓不到;這裡比對整張全景片的影像特徵,把最相似的配對列出來供人工確認。
#   1. 以 ImageNet 預訓練的 ResNet34 取得每張全景片的特徵向量,計算餘弦相似度
#   2. 每張測試全景片列出最相似的訓練全景片,並把配對並排存成圖片
PANO_DIRS = {"first": ROOT / "dental_project" / "original_pictures", "second": ROOT / "dental_project" / "20260826__mf_851"}
OUT_DIR = ROOT / "compare_results" / "same_patient_check"
SIDE = re.compile(r"^(?P<number>\d+)_(Left|Right)_roi$")
TOP_K = 3
REVIEW_PAIRS = 40


def find_pano(batch, number):
    folder = PANO_DIRS[batch]
    stem = str(number) if batch == "first" else f"0827_{number}"
    for path in [folder / stem] + sorted(folder.glob(f"{stem}.*")):
        if path.is_file():
            return path
    return None


@torch.inference_mode()
def embed(paths):
    net = torchvision.models.resnet34(weights="IMAGENET1K_V1")
    net.fc = torch.nn.Identity()
    net = net.to(DEVICE).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
    features = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        # 全景片長寬比約 2:1;統一大小並做直方圖等化,降低不同機器、曝光的影響
        image = cv2.equalizeHist(cv2.resize(image, (512, 256), interpolation=cv2.INTER_AREA))
        x = torch.from_numpy(image).float().div(255).to(DEVICE)[None, None].repeat(1, 3, 1, 1)
        f = net((x - mean) / std)[0]
        features.append(torch.nn.functional.normalize(f, dim=0).cpu().numpy())
    return np.stack(features)


def side_by_side(a, b, title):
    images = []
    for path in (a, b):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        images.append(cv2.resize(image, (900, int(900 * image.shape[0] / image.shape[1]))))
    height = max(i.shape[0] for i in images)
    images = [cv2.copyMakeBorder(i, 0, height - i.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0) for i in images]
    canvas = cv2.cvtColor(np.hstack([images[0], np.zeros((height, 10), np.uint8), images[1]]), cv2.COLOR_GRAY2BGR)
    header = np.zeros((34, canvas.shape[1], 3), np.uint8)
    cv2.putText(header, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([header, canvas])


def main():
    # 訓練資料的全景片:使用 locate_roi_sources.py 以影像內容找到的真正來源(第一批的編號與 original_pictures 的檔名不對應)
    sources_csv = ROOT / "compare_results" / "same_patient_check" / "roi_sources.csv"
    if not sources_csv.exists():
        raise SystemExit(f"找不到 {sources_csv},請先執行 locate_roi_sources.py")
    with open(sources_csv, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["set"] == "train" and r["source_pano"]]
    with open(PATIENT_MAP, newline="") as f:
        batch_of = {r["stem"]: r["batch"] for r in csv.DictReader(f)}
    pano_root = ROOT / "dental_project" / "original_pictures"
    train_panos = {}
    for r in rows:
        if float(r["score"]) >= 0.9 and float(r["score"]) - float(r["second_score"] or 0) >= 0.06:
            train_panos.setdefault((batch_of[r["stem"]], r["pano_group"]), pano_root / r["pano_group"])
    missing = []
    # 測試集來自第二批的來源資料夾
    test_numbers = sorted({SIDE.match(s)["number"] for s in index_files((ROOT / IMAGES_DIR).resolve()) if SIDE.match(s)})
    test_panos = {n: find_pano("second", n) for n in test_numbers}
    missing += [("test", n) for n, p in test_panos.items() if p is None]
    test_panos = {n: p for n, p in test_panos.items() if p is not None}
    print(f"訓練全景片 {len(train_panos)} 張、測試全景片 {len(test_panos)} 張" + (f";找不到 {len(missing)} 張(前5: {missing[:5]})" if missing else ""))

    train_keys = list(train_panos)
    test_keys = list(test_panos)
    similarity = embed([test_panos[k] for k in test_keys]) @ embed([train_panos[k] for k in train_keys]).T

    # 以全部「測試 × 訓練」配對的相似度分布作為參考,列出每張測試片最相似的幾張
    flat = np.sort(similarity.ravel())
    pairs = []
    for i, test_key in enumerate(test_keys):
        for j in np.argsort(-similarity[i])[:TOP_K]:
            percentile = np.searchsorted(flat, similarity[i, j]) / len(flat)
            pairs.append((float(similarity[i, j]), percentile, test_key, train_keys[j]))
    pairs.sort(reverse=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    review = OUT_DIR / "review"
    review.mkdir(exist_ok=True)
    with open(OUT_DIR / "candidates.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "similarity", "percentile", "test_pano", "train_batch", "train_number", "train_pano"])
        for rank, (s, pct, test_key, (batch, number)) in enumerate(pairs, start=1):
            writer.writerow([rank, round(s, 4), round(pct, 5), test_panos[test_key].name, batch, number, train_panos[(batch, number)].name])
    for rank, (s, pct, test_key, key) in enumerate(pairs[:REVIEW_PAIRS], start=1):
        title = f"#{rank}  similarity {s:.3f}  left: test {test_panos[test_key].name}   right: train ({key[0]}) {train_panos[key].name}"
        cv2.imwrite(str(review / f"{rank:03d}_{s:.3f}.jpg"), side_by_side(test_panos[test_key], train_panos[key], title))

    print(f"\n相似度最高的 {REVIEW_PAIRS} 組配對已並排存到: {review}")
    print(f"全部候選(每張測試片前 {TOP_K} 名): {OUT_DIR / 'candidates.csv'}")
    print("相似度的參考分布: 中位數 {:.3f}、第 99 百分位 {:.3f}、最大 {:.3f}".format(np.median(flat), np.percentile(flat, 99), flat[-1]))
    print("請從編號 001 開始檢視,判斷左右兩張是否為同一個人(牙齒排列、修復物、下顎形狀);確認是同一人的請記下編號")


if __name__ == "__main__":
    main()
