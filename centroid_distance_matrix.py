import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics.models.yolo.segment import SegmentationPredictor

from mask_threshold__grid_search import (
    get_raw_prediction_masks,
    merge_prediction_masks,
)


MODEL_PATH = (
    "runs/segment/20260903/yolov8n_mf_341_unsharp_mask_brightness_0.2/weights/best.pt"
)
VAL_IMAGES_DIR = Path(
    "dental_project/20260903/20260903_mf_341_unsharp_mask_brightness_0.2/images"
)
VAL_LABELS_DIR = Path(
    "dental_project/20260903/20260903_mf_341_unsharp_mask_brightness_0.2/labels"
)
OUTPUT_CSV = Path("runs/segment/341_results.csv")
TUNING_OUTPUT_CSV = Path("runs/segment/341_conf_iou_search.csv")
MF_CLASS_ID = 0
MASK_THRESHOLD = 0.56
CONFIDENCE_THRESHOLD = 0.45  # 提高信心度，過濾掉骨小樑弱陰影
IOU_THRESHOLD = 0.30         # 嚴格 NMS，壓制同一孔周圍的重複框
PIXEL_SPACING_MM = 0.1
RUN_PARAMETER_SEARCH = False
MINIMUM_COVERAGE = 0.90
CONFIDENCE_CANDIDATES = np.arange(0.15, 0.56, 0.05)
IOU_CANDIDATES = np.arange(0.30, 0.91, 0.10)


def get_mask_centroid(mask):
    """計算 binary mask 的幾何重心，回傳 (x, y)。"""
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        return None

    return np.array(
        [
            moments["m10"] / moments["m00"],
            moments["m01"] / moments["m00"],
        ],
        dtype=np.float64,
    )


def get_best_single_mask(raw_masks, threshold=MASK_THRESHOLD):
    """
    針對單側 ROI 影像，僅保留 Confidence 最高或面積最大且形狀最完整的單一 Mask，
    避免將多個假陽性融合拉偏質心。
    """
    if raw_masks is None or len(raw_masks) == 0:
        return None
    
    # 若 raw_masks 包含信心度排序，取 confidence 最高的第 0 個
    # 否則對二值化後的結果取最大連通區域 (Largest Connected Component)
    if isinstance(raw_masks, list):
        primary_mask = raw_masks[0] # 取 top-1 預測
    else:
        primary_mask = raw_masks
        
    binary_mask = (primary_mask >= threshold).astype(np.uint8)
    
    # 連通域分析：只保留面積最大的主孔，濾除零碎噪點
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)
    if num_labels <= 1:
        return None
    
    # 忽略背景 (label 0)，取面積最大者
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cleaned_mask = (labels == largest_label).astype(np.uint8)
    
    return cleaned_mask


def read_ground_truth_mask(label_path, width, height):
    """讀取 YOLO segmentation polygon，建立 mf 的 GT mask。"""
    ground_truth = np.zeros((height, width), dtype=np.uint8)

    with label_path.open("r", encoding="utf-8") as label_file:
        for line in label_file:
            values = line.strip().split()
            if len(values) < 7 or int(values[0]) != MF_CLASS_ID:
                continue

            points = np.asarray(
                [float(value) for value in values[1:]],
                dtype=np.float32,
            )
            if len(points) % 2 != 0:
                continue

            points = points.reshape(-1, 2)
            points[:, 0] *= width
            points[:, 1] *= height
            polygon = np.rint(points).astype(np.int32)
            cv2.fillPoly(ground_truth, [polygon], 1)

    return ground_truth


def create_predictor(model_path, confidence_threshold, iou_threshold):
    predictor = SegmentationPredictor(
        overrides={
            "model": model_path,
            "conf": confidence_threshold,
            "iou": iou_threshold,
            "imgsz": 1280,
            "retina_masks": True,
            "verbose": False,
        }
    )
    predictor.setup_model(model_path, verbose=False)
    predictor.imgsz = 1280
    return predictor


def score_parameter_pair(
    model_path,
    val_img_dir,
    val_lbl_dir,
    mask_threshold,
    confidence_threshold,
    iou_threshold,
):
    """評估一組 conf/IoU，回傳誤差與涵蓋率，不輸出逐張 CSV。"""
    predictor = create_predictor(
        model_path,
        confidence_threshold,
        iou_threshold,
    )
    valid_ground_truths = 0
    successful_predictions = 0
    errors_px = []

    for image_path in sorted(Path(val_img_dir).glob("*.jpg")):
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        height, width = image.shape[:2]
        label_path = Path(val_lbl_dir) / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        ground_truth = read_ground_truth_mask(label_path, width, height)
        gt_mask = get_best_single_mask(ground_truth)
        if gt_mask is None:
            continue

        gt_centroid = get_mask_centroid(gt_mask)
        if gt_centroid is None:
            continue
        valid_ground_truths += 1

        raw_masks = get_raw_prediction_masks(predictor, image)
        prediction_mask = get_best_single_mask(
            raw_masks,
            threshold=mask_threshold,
        )
        if prediction_mask is None:
            continue

        pred_centroid = get_mask_centroid(prediction_mask)
        if pred_centroid is None:
            continue

        errors_px.append(float(np.linalg.norm(gt_centroid - pred_centroid)))
        successful_predictions += 1

    errors = np.asarray(errors_px, dtype=np.float64)
    coverage = (
        successful_predictions / valid_ground_truths
        if valid_ground_truths
        else 0.0
    )
    return {
        "confidence": float(confidence_threshold),
        "iou": float(iou_threshold),
        "valid_ground_truths": valid_ground_truths,
        "successful_predictions": successful_predictions,
        "coverage": coverage,
        "mean_error_px": float(np.mean(errors)) if len(errors) else np.inf,
        "median_error_px": float(np.median(errors)) if len(errors) else np.inf,
    }


def search_best_conf_iou(
    model_path,
    val_img_dir,
    val_lbl_dir,
    output_csv,
    mask_threshold=MASK_THRESHOLD,
    confidence_candidates=CONFIDENCE_CANDIDATES,
    iou_candidates=IOU_CANDIDATES,
    minimum_coverage=MINIMUM_COVERAGE,
):
    """搜尋 conf/IoU，避免用漏檢換取看似較低的平均誤差。"""
    search_results = []
    for confidence in confidence_candidates:
        for iou in iou_candidates:
            print(f"Testing conf={confidence:.2f}, NMS IoU={iou:.2f}")
            search_results.append(
                score_parameter_pair(
                    model_path,
                    val_img_dir,
                    val_lbl_dir,
                    mask_threshold,
                    float(confidence),
                    float(iou),
                )
            )

    qualified = [
        result
        for result in search_results
        if result["coverage"] >= minimum_coverage
    ]
    if qualified:
        best = min(
            qualified,
            key=lambda result: (
                result["mean_error_px"],
                -result["coverage"],
            ),
        )
    else:
        best = max(
            search_results,
            key=lambda result: (
                result["coverage"],
                -result["mean_error_px"],
            ),
        )

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(search_results[0].keys()) if search_results else []
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(search_results)

    print("=== Best conf/IoU parameters ===")
    print(f"Minimum required coverage: {minimum_coverage:.2%}")
    print(f"Confidence threshold: {best['confidence']:.2f}")
    print(f"NMS IoU threshold: {best['iou']:.2f}")
    print(f"Coverage: {best['coverage']:.2%}")
    print(f"Mean centroid error: {best['mean_error_px']:.2f} px")
    print(f"Median centroid error: {best['median_error_px']:.2f} px")
    print(f"Search results saved to: {output_path}")
    return best


def evaluate_centroid_errors(
    model_path,
    val_img_dir,
    val_lbl_dir,
    output_csv,
    pixel_spacing_mm=PIXEL_SPACING_MM,
    mask_threshold=MASK_THRESHOLD,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    iou_threshold=IOU_THRESHOLD,
):
    predictor = create_predictor(
        model_path,
        confidence_threshold,
        iou_threshold,
    )
    results = []
    missing_labels = 0
    missing_predictions = 0
    invalid_ground_truth = 0

    for image_path in sorted(Path(val_img_dir).glob("*.jpg")):
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        height, width = image.shape[:2]
        label_path = Path(val_lbl_dir) / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_labels += 1
            continue

        ground_truth = read_ground_truth_mask(label_path, width, height)
        gt_mask = get_best_single_mask(ground_truth)
        if gt_mask is None:
            invalid_ground_truth += 1
            continue
        gt_centroid = get_mask_centroid(gt_mask)
        if gt_centroid is None:
            invalid_ground_truth += 1
            continue

        raw_masks = get_raw_prediction_masks(predictor, image)
        prediction_mask = get_best_single_mask(
            raw_masks,
            threshold=mask_threshold,
        )

        if prediction_mask is None:
            missing_predictions += 1
            continue

        pred_centroid = get_mask_centroid(prediction_mask)
        if pred_centroid is None:
            missing_predictions += 1
            continue

        distance_px = float(np.linalg.norm(gt_centroid - pred_centroid))
        distance_mm = distance_px * pixel_spacing_mm
        results.append(
            {
                "image": image_path.name,
                "gt_x": gt_centroid[0],
                "gt_y": gt_centroid[1],
                "pred_x": pred_centroid[0],
                "pred_y": pred_centroid[1],
                "error_px": distance_px,
                "error_mm": distance_mm,
            }
        )

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=results[0].keys() if results else [
            "image", "gt_x", "gt_y", "pred_x", "pred_y", "error_px", "error_mm"
        ])
        writer.writeheader()
        writer.writerows(results)

    errors_px = np.asarray([row["error_px"] for row in results])
    errors_mm = np.asarray([row["error_mm"] for row in results])

    print("=== Centroid Localization Evaluation ===")
    print(f"Mask threshold: {mask_threshold:.2f}")
    print(f"Confidence threshold: {confidence_threshold:.2f}")
    print(f"NMS IoU threshold: {iou_threshold:.2f}")
    print(f"Validated instances: {len(results)}")
    print(f"Missing predictions: {missing_predictions}")
    print(f"Missing labels: {missing_labels}")
    print(f"Invalid ground truth masks: {invalid_ground_truth}")
    if len(results) > 0:
        print(
            f"Mean error: {np.mean(errors_px):.2f} +/- "
            f"{np.std(errors_px):.2f} px"
        )
        print(
            f"Mean physical error: {np.mean(errors_mm):.2f} +/- "
            f"{np.std(errors_mm):.2f} mm"
        )
        print(f"Median error: {np.median(errors_mm):.2f} mm")
        print(f"Maximum error: {np.max(errors_mm):.2f} mm")
    else:
        print("No valid centroid results were produced.")
    print(f"Per-image results saved to: {output_path}")


if __name__ == "__main__":
    if RUN_PARAMETER_SEARCH:
        best_parameters = search_best_conf_iou(
            model_path=MODEL_PATH,
            val_img_dir=VAL_IMAGES_DIR,
            val_lbl_dir=VAL_LABELS_DIR,
            output_csv=TUNING_OUTPUT_CSV,
        )
        CONFIDENCE_THRESHOLD = best_parameters["confidence"]
        IOU_THRESHOLD = best_parameters["iou"]

    evaluate_centroid_errors(
        model_path=MODEL_PATH,
        val_img_dir=VAL_IMAGES_DIR,
        val_lbl_dir=VAL_LABELS_DIR,
        output_csv=OUTPUT_CSV,
        pixel_spacing_mm=PIXEL_SPACING_MM,
        mask_threshold=MASK_THRESHOLD,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
    )
# 執行評估
# evaluate_centroid_errors("runs/segment/your_exp/weights/best.pt", "path/to/val/images", "path/to/val/labels")