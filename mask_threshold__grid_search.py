import csv

import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics.models.yolo.detect import predict as detect_predict
from ultralytics.models.yolo.segment import SegmentationPredictor
from ultralytics.utils import ops


MODEL_PATH = "runs/segment/20260903/yolov8n_mf_341_unsharp_mask_brightness_0.2/weights/best.pt"
VAL_IMAGES_DIR = Path(
    "datasets/20260903/20260903_mf_341_unsharp_mask_brightness_0.2/valid/images"
)
VAL_LABELS_DIR = Path(
    "datasets/20260903/20260903_mf_341_unsharp_mask_brightness_0.2/valid/labels"
)
MF_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.20
IOU_THRESHOLD = 0.30
MASK_THRESHOLDS = np.arange(0.10, 0.91, 0.01)
RUN_CONF_IOU_SEARCH = False
CONFIDENCE_CANDIDATES = np.arange(0.15, 0.56, 0.05)
IOU_CANDIDATES = np.arange(0.30, 0.91, 0.10)
CONF_IOU_SEARCH_OUTPUT = Path("runs/segment/mask_conf_iou_search.csv")


def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 1.0
    return intersection / union


def refine_mask_with_ellipse(pred_mask_binary):
    """以最大外輪廓擬合實心橢圓，無法擬合時保留原始 mask。"""
    contours, _ = cv2.findContours(
        pred_mask_binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return pred_mask_binary

    largest_contour = max(contours, key=cv2.contourArea)
    if len(largest_contour) < 5:
        return pred_mask_binary

    try:
        ellipse = cv2.fitEllipse(largest_contour)
    except cv2.error:
        return pred_mask_binary

    refined_mask = np.zeros_like(pred_mask_binary)
    cv2.ellipse(refined_mask, ellipse, color=1, thickness=-1)
    return refined_mask


def read_ground_truth_mask(label_path, width, height):
    """讀取 YOLO segmentation polygon，合併成指定類別的 binary mask。"""
    ground_truth = np.zeros((height, width), dtype=np.uint8)

    with label_path.open("r", encoding="utf-8") as label_file:
        for line in label_file:
            values = line.strip().split()
            if len(values) < 7:
                continue

            class_id = int(values[0])
            if class_id != MF_CLASS_ID:
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


def get_raw_prediction_masks(predictor, image):
    """取得原始連續 mask probability，而不是 Ultralytics 的 binary mask。"""
    height, width = image.shape[:2]
    input_tensor = predictor.preprocess([image])

    with torch.inference_mode():
        model_output = predictor.model(input_tensor)

    raw_predictions, prototypes = model_output[0]
    raw_predictions = raw_predictions.clone()
    detections = detect_predict.nms.non_max_suppression(
        raw_predictions,
        predictor.args.conf,
        predictor.args.iou,
        predictor.args.classes,
        predictor.args.agnostic_nms,
        max_det=predictor.args.max_det,
        nc=len(predictor.model.names),
        end2end=getattr(predictor.model, "end2end", False),
        rotated=False,
    )[0]

    if detections.shape[0] == 0:
        return []

    class_mask = detections[:, 5].int() == MF_CLASS_ID
    detections = detections[class_mask]
    if detections.shape[0] == 0:
        return []

    boxes = detections[:, :4].clone()
    boxes = ops.scale_boxes(
        input_tensor.shape[2:],
        boxes,
        image.shape[:2],
    )

    coefficients = detections[:, 6:]
    prototype = prototypes[0]
    channels, mask_height, mask_width = prototype.shape
    logits = (
        coefficients @ prototype.float().view(channels, -1)
    ).view(-1, mask_height, mask_width)

    # Match Ultralytics retina_masks coordinate handling, but keep logits.
    logits = ops.scale_masks(
        logits[None],
        (height, width),
    )[0]
    logits = ops.crop_mask(logits, boxes)
    return [mask.sigmoid().cpu().numpy() for mask in logits]


def merge_prediction_masks(raw_masks, threshold, refine):
    """逐一處理 prediction instance，再合併成單張 mask。"""
    if not raw_masks:
        return None

    height, width = raw_masks[0].shape
    merged_mask = np.zeros((height, width), dtype=np.uint8)

    for raw_mask in raw_masks:
        binary_mask = (raw_mask > threshold).astype(np.uint8)
        if refine:
            binary_mask = refine_mask_with_ellipse(binary_mask)
        merged_mask = np.maximum(merged_mask, binary_mask)

    return merged_mask


def load_evaluation_samples():
    """預先讀取有標註的影像與 GT mask，避免每組參數重複讀檔。"""
    samples = []
    missing_labels = 0

    for image_path in sorted(VAL_IMAGES_DIR.glob("*.jpg")):
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        label_path = VAL_LABELS_DIR / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_labels += 1
            continue

        height, width = image.shape[:2]
        ground_truth = read_ground_truth_mask(label_path, width, height)
        samples.append((image_path.name, image, ground_truth))

    return samples, missing_labels


def search_conf_iou_by_mask_iou():
    """以 mask IoU 搜尋最佳 conf、NMS IoU 與 mask threshold。"""
    samples, missing_labels = load_evaluation_samples()
    search_rows = []

    for confidence in CONFIDENCE_CANDIDATES:
        for nms_iou in IOU_CANDIDATES:
            print(
                f"Testing conf={confidence:.2f}, "
                f"NMS IoU={nms_iou:.2f}"
            )
            predictor = SegmentationPredictor(
                overrides={
                    "model": MODEL_PATH,
                    "conf": float(confidence),
                    "iou": float(nms_iou),
                    "imgsz": 640,
                    "retina_masks": True,
                    "verbose": False,
                }
            )
            predictor.setup_model(MODEL_PATH, verbose=False)
            predictor.imgsz = 640

            raw_iou_values = {round(float(t), 2): [] for t in MASK_THRESHOLDS}
            refined_iou_values = {
                round(float(t), 2): [] for t in MASK_THRESHOLDS
            }

            for _, image, ground_truth in samples:
                prediction_masks = get_raw_prediction_masks(predictor, image)
                for threshold in MASK_THRESHOLDS:
                    threshold_key = round(float(threshold), 2)
                    raw_prediction = merge_prediction_masks(
                        prediction_masks,
                        threshold,
                        refine=False,
                    )
                    refined_prediction = merge_prediction_masks(
                        prediction_masks,
                        threshold,
                        refine=True,
                    )

                    if raw_prediction is None:
                        raw_prediction = np.zeros_like(ground_truth)
                        refined_prediction = np.zeros_like(ground_truth)

                    raw_iou_values[threshold_key].append(
                        calculate_iou(raw_prediction, ground_truth)
                    )
                    refined_iou_values[threshold_key].append(
                        calculate_iou(refined_prediction, ground_truth)
                    )

            best_raw_threshold = max(
                raw_iou_values,
                key=lambda threshold: np.mean(raw_iou_values[threshold]),
            )
            best_refined_threshold = max(
                refined_iou_values,
                key=lambda threshold: np.mean(
                    refined_iou_values[threshold]
                ),
            )
            search_rows.append(
                {
                    "confidence": float(confidence),
                    "nms_iou": float(nms_iou),
                    "best_raw_threshold": best_raw_threshold,
                    "best_raw_mean_iou": float(
                        np.mean(raw_iou_values[best_raw_threshold])
                    ),
                    "best_refined_threshold": best_refined_threshold,
                    "best_refined_mean_iou": float(
                        np.mean(
                            refined_iou_values[best_refined_threshold]
                        )
                    ),
                    "evaluated_images": len(samples),
                    "missing_labels": missing_labels,
                }
            )

    output_path = CONF_IOU_SEARCH_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(search_rows[0].keys()) if search_rows else []
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(search_rows)

    best_raw = max(search_rows, key=lambda row: row["best_raw_mean_iou"])
    best_refined = max(
        search_rows,
        key=lambda row: row["best_refined_mean_iou"],
    )
    print("=== Best parameters by Original mask IoU ===")
    print(f"Confidence threshold: {best_raw['confidence']:.2f}")
    print(f"NMS IoU threshold: {best_raw['nms_iou']:.2f}")
    print(f"Mask threshold: {best_raw['best_raw_threshold']:.2f}")
    print(f"Mean IoU: {best_raw['best_raw_mean_iou']:.4f}")
    print("=== Best parameters by Ellipse mask IoU ===")
    print(f"Confidence threshold: {best_refined['confidence']:.2f}")
    print(f"NMS IoU threshold: {best_refined['nms_iou']:.2f}")
    print(f"Mask threshold: {best_refined['best_refined_threshold']:.2f}")
    print(f"Mean IoU: {best_refined['best_refined_mean_iou']:.4f}")
    print(f"Search results saved to: {output_path}")


def main():
    predictor = SegmentationPredictor(
        overrides={
            "model": MODEL_PATH,
            "conf": CONFIDENCE_THRESHOLD,
            "iou": IOU_THRESHOLD,
            "imgsz": 640,
            "retina_masks": True,
            "verbose": False,
        }
    )
    predictor.setup_model(MODEL_PATH, verbose=False)
    predictor.imgsz = 640
    raw_ious = {round(float(t), 2): [] for t in MASK_THRESHOLDS}
    refined_ious = {round(float(t), 2): [] for t in MASK_THRESHOLDS}
    evaluated_images = 0
    missing_labels = 0

    for image_path in sorted(VAL_IMAGES_DIR.glob("*.jpg")):
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        height, width = image.shape[:2]
        label_path = VAL_LABELS_DIR / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_labels += 1
            continue

        ground_truth = read_ground_truth_mask(label_path, width, height)
        prediction_masks = get_raw_prediction_masks(predictor, image)

        for threshold in MASK_THRESHOLDS:
            threshold_key = round(float(threshold), 2)
            raw_prediction = merge_prediction_masks(
                prediction_masks,
                threshold,
                refine=False,
            )
            refined_prediction = merge_prediction_masks(
                prediction_masks,
                threshold,
                refine=True,
            )

            if raw_prediction is None:
                raw_prediction = np.zeros_like(ground_truth)
                refined_prediction = np.zeros_like(ground_truth)

            raw_ious[threshold_key].append(
                calculate_iou(raw_prediction, ground_truth)
            )
            refined_ious[threshold_key].append(
                calculate_iou(refined_prediction, ground_truth)
            )

        evaluated_images += 1

    print("=== Mental Foramen Mask Evaluation ===")
    print(f"Evaluated images: {evaluated_images}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD:.2f}")
    print(f"NMS IoU threshold: {IOU_THRESHOLD:.2f}")
    if missing_labels:
        print(f"Skipped images without labels: {missing_labels}")

    print("\nMask threshold | Original IoU | Ellipse IoU")
    print("---------------|---------------|------------")
    for threshold in raw_ious:
        original_mean = np.mean(raw_ious[threshold]) if raw_ious[threshold] else 0.0
        refined_mean = (
            np.mean(refined_ious[threshold])
            if refined_ious[threshold]
            else 0.0
        )
        print(f"{threshold:9.2f} | {original_mean:13.4f} | {refined_mean:10.4f}")

    best_original = max(raw_ious, key=lambda key: np.mean(raw_ious[key]))
    best_refined = max(refined_ious, key=lambda key: np.mean(refined_ious[key]))
    print(f"\nBest original threshold: {best_original:.2f}")
    print(f"Best original mean IoU: {np.mean(raw_ious[best_original]):.4f}")
    print(f"Best ellipse threshold: {best_refined:.2f}")
    print(f"Best ellipse mean IoU: {np.mean(refined_ious[best_refined]):.4f}")


if __name__ == "__main__":
    if RUN_CONF_IOU_SEARCH:
        search_conf_iou_by_mask_iou()
    else:
        main()