import argparse
import csv
import re
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from stage2.compare_yolo_unet import GT_DIR, IMAGES_DIR, NAMES, ROOT, index_files, load_gt
from stage2.kfold_pano import EXCLUDE_TEST, PROJECT, SUMMARY_KEYS, evaluate
from stage2.kfold_patient import save_masks
from stage2.kfold_unet_variants import CONFIGS, build_model, fold_paths
from stage2.kfold_yolo_unet import K_FOLDS
from stage2.predict_unet import DEVICE

# 推論時左右翻轉(TTA,2026-10-28): 五折多數決比單一模型多約 4–5 點測試命中,但運算是 5 倍。
#   用已訓練好的 kfold_pano U-Net(訓練時本來就有隨機左右翻轉增強),每張影像推論「原圖 + 左右翻轉」兩次,
#   softmax 平均後取 argmax(2 倍運算),看能不能用比五折少的運算拿到接近的命中率。
#   遮罩存到 masks/{cv,test}/<設定>_tta/,評分與 kfold_pano.evaluate 相同(CV 每張用沒看過它的那一折;測試集單一模型平均與五折多數決)。
#   注意: 原本的推論先內插 logits 再 argmax,這裡內插的是平均後的機率;差異只在邊界的少數像素。
TAGS = ["r34_imagenet_aug"] + [f"r34_imagenet_aug_s{s}" for s in range(1, 5)] \
    + ["r34_imagenet_aug_384"] + [f"r34_imagenet_aug_384_s{s}" for s in range(1, 5)]


@torch.inference_mode()
def predict_tta(model, image, size):
    # 回傳原圖解析度的 {類別: bool 遮罩}
    h, w = image.shape
    x = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    x = torch.from_numpy(x)[None, None].to(DEVICE)
    prob = model(x).softmax(1) + model(x.flip(-1)).softmax(1).flip(-1)
    pred = F.interpolate(prob, size=(h, w), mode="bilinear", align_corners=False).argmax(1)[0].cpu().numpy()
    return {cls: pred == c + 1 for c, cls in NAMES.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", default=TAGS, help="kfold_pano 的 U-Net 名稱(設定名稱,種子不是 0 時加 _s<種子>)")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    out_dir = ROOT / "compare_results" / PROJECT
    test_images = {s: p for s, p in index_files((ROOT / IMAGES_DIR).resolve()).items() if s not in EXCLUDE_TEST}
    test_gt = (ROOT / GT_DIR).resolve()
    test = []
    for stem, path in test_images.items():
        shape = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).shape
        test.append((stem, load_gt(stem, shape, test_gt, {})))

    summary = []
    for tag in args.tags:
        config = CONFIGS[re.sub(r"_s\d+$", "", tag)]
        folds = [f for f in range(K_FOLDS) if (out_dir / "weights" / f"{tag}_fold{f}.pth").exists()]
        if len(folds) != K_FOLDS:
            print(f"[略過] {tag}: 只有 {len(folds)} 折的權重")
            continue
        for fold in folds:
            unet = build_model({**config, "pretrained": False}).to(DEVICE)
            unet.load_state_dict(torch.load(out_dir / "weights" / f"{tag}_fold{fold}.pth", map_location=DEVICE)["model_state_dict"])
            unet.eval()
            for dataset, paths in (("cv", fold_paths(fold, "val", out_dir)), ("test", list(test_images.values()))):
                for path in paths:
                    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    save_masks(predict_tta(unet, image, config["size"]), out_dir / "masks" / dataset / f"{tag}_tta" / f"fold{fold}" / f"{path.stem}.png")
            del unet
            torch.cuda.empty_cache()
        print(f"[完成推論] {tag}")
        summary += evaluate(tag, folds, out_dir, test) + evaluate(f"{tag}_tta", folds, out_dir, test)

    print(f"\n{'model':<32}{'setting':<16}{'n':>5}" + "".join(f"{k:>14}" for k in SUMMARY_KEYS))
    for r in summary:
        print(f"{r['model']:<32}{r['setting']:<16}{r['n']:>5}" + "".join(f"{r[k]:>14.4f}" for k in SUMMARY_KEYS))
    with open(out_dir / "summary_tta.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"\n彙總結果: {out_dir / 'summary_tta.csv'}")


if __name__ == "__main__":
    main()
