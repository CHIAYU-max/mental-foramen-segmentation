from pathlib import Path

import cv2
import numpy as np


PREDICTION_DIR = Path("runs/segment/predict")
LABEL_DIR = Path(
    "datasets/20260903/20260903_mf_341_unsharp_mask_brightness_0.2/valid/labels"
)
OUTPUT_DIR = Path("runs/segment/predict_overlay")
GROUND_TRUTH_COLOR = (0, 0, 255)
FILL_ALPHA = 0.18


def read_polygons(label_path, width, height):
    polygons = []
    for line in label_path.read_text().splitlines():
        values = line.split()
        if len(values) < 7 or (len(values) - 1) % 2:
            continue

        class_id = int(values[0])
        points = np.asarray([float(value) for value in values[1:]], dtype=np.float32)
        points[0::2] *= width
        points[1::2] *= height
        polygon = np.rint(points.reshape(-1, 2)).astype(np.int32)
        polygons.append((class_id, polygon))
    return polygons


def overlay_ground_truth(image, polygons):
    overlay = image.copy()
    for _, polygon in polygons:
        cv2.fillPoly(overlay, [polygon], GROUND_TRUTH_COLOR)
    image = cv2.addWeighted(overlay, FILL_ALPHA, image, 1 - FILL_ALPHA, 0)

    for class_id, polygon in polygons:
        cv2.polylines(image, [polygon], True, GROUND_TRUTH_COLOR, 2, cv2.LINE_AA)
        x, y = polygon[0]
        cv2.putText(
            image,
            f"GT {class_id}",
            (int(x), max(int(y) - 5, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            GROUND_TRUTH_COLOR,
            1,
            cv2.LINE_AA,
        )
    return image


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prediction_images = sorted(PREDICTION_DIR.glob("*.jpg"))
    missing_labels = []

    for prediction_path in prediction_images:
        label_path = LABEL_DIR / f"{prediction_path.stem}.txt"
        if not label_path.exists():
            missing_labels.append(prediction_path.name)
            continue

        image = cv2.imread(str(prediction_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        polygons = read_polygons(label_path, width, height)
        result = overlay_ground_truth(image, polygons)
        cv2.imwrite(str(OUTPUT_DIR / prediction_path.name), result)

    print(f"Wrote {len(list(OUTPUT_DIR.glob('*.jpg')))} images to {OUTPUT_DIR}")
    if missing_labels:
        print(f"Missing labels: {', '.join(missing_labels)}")


if __name__ == "__main__":
    main()