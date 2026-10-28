import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def load_gt_from_txt(txt_path: str, img_shape: tuple) -> np.ndarray:
    """將 YOLO 格式的 txt 多邊形標註轉換為二值遮罩 (H, W)"""
    h, w = img_shape[:2]
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    if not os.path.exists(txt_path):
        return gt_mask

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        # 提取多邊形座標並反歸一化為像素座標
        coords = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2)
        coords[:, 0] *= w
        coords[:, 1] *= h
        pts = coords.astype(np.int32)
        cv2.fillPoly(gt_mask, [pts], color=1)
    return gt_mask

def evaluate_yolo_semantic_metrics(
    model_path: str,
    images_dir: str,
    labels_dir: str,
    conf_thresh: float = 0.25,
    gt_type: str = "txt"  # "txt" 或 "mask" (png/bmp)
):
    """
    評估 YOLO 模型在測試集上的像素級指標 (Dice, Precision, Recall, mIoU)
    """
    model = YOLO(model_path)
    img_paths = list(Path(images_dir).glob("*.*"))
    img_paths = [p for p in img_paths if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]

    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    per_image_dice = []
    per_image_iou = []

    print(f"開始評估 {len(img_paths)} 張測試影像...")

    for img_path in img_paths:
        img_bgr = cv2.imread(str(img_path))
        h, w = img_bgr.shape[:2]

        # 1. 取得 Ground Truth 二值遮罩
        if gt_type == "txt":
            label_file = Path(labels_dir) / f"{img_path.stem}.txt"
            gt_mask = load_gt_from_txt(str(label_file), (h, w))
        else:
            # 支援 .png 或同名遮罩
            mask_file = Path(labels_dir) / f"{img_path.stem}.png"
            if not mask_file.exists():
                mask_file = Path(labels_dir) / f"{img_path.stem}.bmp"
            
            if mask_file.exists():
                gt_raw = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                gt_mask = (gt_raw > 127).astype(np.uint8)
            else:
                gt_mask = np.zeros((h, w), dtype=np.uint8)

        # 2. 執行 YOLO 推論並合併所有實例遮罩為單一語意遮罩
        results = model.predict(source=str(img_path), conf=conf_thresh, verbose=False)
        pred_mask = np.zeros((h, w), dtype=np.uint8)

        if results[0].masks is not None:
            # results[0].masks.data 包含所有偵測到的實例遮罩
            raw_masks = results[0].masks.data.cpu().numpy()  # shape: (N, mask_h, mask_w)
            for m in raw_masks:
                # 縮放回原始影像尺寸
                resized_m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                pred_mask = np.bitwise_or(pred_mask, (resized_m > 0.5).astype(np.uint8))

        # 3. 統計像素混淆矩陣
        tp = np.sum((pred_mask == 1) & (gt_mask == 1))
        fp = np.sum((pred_mask == 1) & (gt_mask == 0))
        fn = np.sum((pred_mask == 0) & (gt_mask == 1))

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # 逐張圖片計算（防止除以 0）
        eps = 1e-7
        img_dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
        img_iou = tp / (tp + fp + fn + eps)

        # 若該圖 GT 與預測皆為空，視為完全匹配
        if np.sum(gt_mask) == 0 and np.sum(pred_mask) == 0:
            img_dice = 1.0
            img_iou = 1.0

        per_image_dice.append(img_dice)
        per_image_iou.append(img_iou)

    # 4. 全域彙總指標（Global / Micro Metrics）
    eps = 1e-7
    global_precision = total_tp / (total_tp + total_fp + eps)
    global_recall = total_tp / (total_tp + total_fn + eps)
    global_dice = (2.0 * total_tp) / (2.0 * total_tp + total_fp + total_fn + eps)
    global_iou = total_tp / (total_tp + total_fp + total_fn + eps)

    # 5. 輸出評估報告
    print("\n" + "=" * 45)
    print("       YOLO 像素層級分割評估結果")
    print("=" * 45)
    print(f"評估影像總數         : {len(img_paths)}")
    print(f"Pixel Precision      : {global_precision * 100:.2f} %")
    print(f"Pixel Recall         : {global_recall * 100:.2f} %")
    print(f"Global Dice (F1)     : {global_dice * 100:.2f} %")
    print(f"Mean Dice (Macro)    : {np.mean(per_image_dice) * 100:.2f} %")
    print(f"Global IoU (mIoU)    : {global_iou * 100:.2f} %")
    print(f"Mean IoU (Macro)     : {np.mean(per_image_iou) * 100:.2f} %")
    print("=" * 45)

    return {
        "precision": global_precision,
        "recall": global_recall,
        "global_dice": global_dice,
        "mean_dice": float(np.mean(per_image_dice)),
        "mean_iou": float(np.mean(per_image_iou))
    }

if __name__ == "__main__":
    # 配置路徑
    MODEL_WEIGHTS = "runs/segment/train/weights/best.pt"
    TEST_IMAGES = "dataset/images/test"
    TEST_LABELS = "dataset/labels/test"  # 或放置 mask png 的資料夾

    evaluate_yolo_semantic_metrics(
        model_path=MODEL_WEIGHTS,
        images_dir=TEST_IMAGES,
        labels_dir=TEST_LABELS,
        conf_thresh=0.25,
        gt_type="txt"  # 若標註是 png 遮罩請改為 "mask"
    )