"""2026-11-14 泛化性的壓力測試(使用者要求;伺服器 CPU,不需要 GPU)。

沒有外部資料的頦孔標註,無法直接量泛化;這裡對測試集 ROI 做合成的畫質變化,看部署的第二階段
(R34/512 OpenVINO INT8 串接式聯集 + lcc,5 種子)在哪些變化下命中掉最多,並看一個不用重訓的補救:
推論前把影像線性調成各折訓練影像的平均 / 標準差(zscore,quality_methods.py;原分布下 11/6 已知沒有幫助)。
變化: gamma、對比縮放(以平均值為中心)、亮度平移、高斯雜訊、高斯模糊、降解析度再放大、JPEG 品質。
判讀: 這是合成變化,只能指出模型對哪一類變化敏感,不能代表真實的跨院差異。程度是事先列好的,不依結果調整。
輸出: compare_results/deploy_eval/robust/results.csv、per_roi.csv
"""
import csv
from collections import defaultdict

import cv2
import numpy as np
import torch

import deployment.deploy_pipeline as D
from stage2.compare_yolo_unet import IMAGES_DIR, ROOT, index_files
from analysis.e2e_seeds import SIDE
from analysis.paper_metrics import image_metrics
from analysis.pano_visibility import load_test
from stage2.quality_methods import train_stats, zscore
from analysis.seed_bootstrap import cluster_test

OUT = ROOT / "compare_results" / "deploy_eval" / "robust"
RNG = np.random.default_rng(0)


def f32(img):
    return img.astype(np.float32)


def u8(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def resample(img, s):
    h, w = img.shape
    small = cv2.resize(img, (max(8, round(w * s)), max(8, round(h * s))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def jpeg(img, q):
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_GRAYSCALE)


CONDITIONS = {
    "original": lambda im, r: im,
    "gamma 0.6": lambda im, r: u8(255 * (f32(im) / 255) ** 0.6),
    "gamma 0.8": lambda im, r: u8(255 * (f32(im) / 255) ** 0.8),
    "gamma 1.25": lambda im, r: u8(255 * (f32(im) / 255) ** 1.25),
    "gamma 1.6": lambda im, r: u8(255 * (f32(im) / 255) ** 1.6),
    "contrast x0.5": lambda im, r: u8((f32(im) - im.mean()) * 0.5 + im.mean()),
    "contrast x0.75": lambda im, r: u8((f32(im) - im.mean()) * 0.75 + im.mean()),
    "contrast x1.5": lambda im, r: u8((f32(im) - im.mean()) * 1.5 + im.mean()),
    "brightness -30": lambda im, r: u8(f32(im) - 30),
    "brightness +30": lambda im, r: u8(f32(im) + 30),
    "noise sd 5": lambda im, r: u8(f32(im) + r.normal(0, 5, im.shape)),
    "noise sd 10": lambda im, r: u8(f32(im) + r.normal(0, 10, im.shape)),
    "blur sd 1": lambda im, r: cv2.GaussianBlur(im, (0, 0), 1),
    "blur sd 2": lambda im, r: cv2.GaussianBlur(im, (0, 0), 2),
    "resolution x0.75": lambda im, r: resample(im, 0.75),
    "resolution x0.5": lambda im, r: resample(im, 0.5),
    "jpeg q50": lambda im, r: jpeg(im, 50),
    "jpeg q30": lambda im, r: jpeg(im, 30),
}


class Normalized:
    # 每一折的模型吃調成該折訓練影像平均 / 標準差的影像
    def __init__(self, model, mu, sd):
        self.model, self.mu, self.sd = model, mu, sd

    def __call__(self, gray):
        return self.model(zscore(gray, self.mu, self.sd))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    D.CASCADE_OUT = "union"
    paths = index_files((ROOT / IMAGES_DIR).resolve())
    test = [(stem, gt, cv2.imread(str(paths[stem]), cv2.IMREAD_GRAYSCALE)) for stem, gt in load_test() if gt["mf"].any()]
    stats = [train_stats(k)[1:] for k in range(5)]
    per = defaultdict(lambda: defaultdict(list))  # (condition, remedy) → stem → [(hit, dice, found, models)]
    with torch.inference_mode():
        for seed in range(5):
            models = D.load_models(seed, 8)
            norm = [Normalized(m, *stats[k]) for k, m in enumerate(models)]
            for name, fn in CONDITIONS.items():
                rng = np.random.default_rng(1000 * seed + 1)
                for stem, gt, img in test:
                    x = fn(img, rng)
                    for remedy, ms in (("none", models), ("zscore", norm)):
                        pred, n = D.cascade(ms, x, D.Clock())
                        m = image_metrics(pred, gt)
                        per[(name, remedy)][stem].append((m["mf_hit_1R"], m["mf_dice"], float(pred["mf"].any()), n))
                print(f"seed {seed} {name}: hit {np.mean([v[-1][0] for v in per[(name, 'none')].values()]):.3f}"
                      f" zscore {np.mean([v[-1][0] for v in per[(name, 'zscore')].values()]):.3f}", flush=True)
    mean = {k: {stem: np.nanmean(np.array(v, float), axis=0) for stem, v in d.items()} for k, d in per.items()}
    ref = mean[("original", "none")]
    rows = []
    for (name, remedy), d in mean.items():
        clusters = {}
        for stem, v in d.items():
            c = SIDE.sub("", stem)
            s0, n0 = clusters.get(c, (0.0, 0))
            clusters[c] = (s0 + v[0] - ref[stem][0], n0 + 1)
        est, lo, hi, p = cluster_test(list(clusters.values()), RNG)
        rows.append({"condition": name, "remedy": remedy, "hit": np.mean([v[0] for v in d.values()]), "hit_diff": est, "ci_low": lo, "ci_high": hi, "p": p,
                     "mf_dice": np.nanmean([v[1] for v in d.values()]), "mf_found": np.mean([v[2] for v in d.values()]),
                     "models_run": np.mean([v[3] for v in d.values()])})
    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "per_roi.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "remedy", "stem", "hit", "mf_dice", "mf_found", "models_run"])
        for (name, remedy), d in mean.items():
            for stem, v in d.items():
                w.writerow([name, remedy, stem, *np.round(v, 5)])
    for r in rows:
        print(f"{r['condition']:<18}{r['remedy']:<8} hit {r['hit']:.4f} ({100 * r['hit_diff']:+.2f} [{100 * r['ci_low']:+.2f}, {100 * r['ci_high']:+.2f}])"
              f" dice {r['mf_dice']:.4f} found {r['mf_found']:.3f} models {r['models_run']:.2f}")


if __name__ == "__main__":
    main()
