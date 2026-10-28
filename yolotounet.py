import os
import cv2
import numpy as np
from pathlib import Path


# ==============================
# 資料夾設定
# ==============================

SOURCE_DATASET = r".\yolodata"

OUTPUT_DATASET = r".\unetdata"


# ==============================
# YOLO class
# ==============================

# 0 = mf = 頦孔
# 1 = ian = 下齒槽神經

CLASS_MAP = {
    0: 1,   # mf → mask value 1
    1: 2    # ian → mask value 2
}


# ==============================
# YOLO Segmentation → Mask
# ==============================

def convert_split(split):

    image_dir = Path(SOURCE_DATASET) / split / "images"
    label_dir = Path(SOURCE_DATASET) / split / "labels"

    output_image_dir = Path(OUTPUT_DATASET) / split / "images"
    output_mask_dir = Path(OUTPUT_DATASET) / split / "masks"

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

    image_files = []

    for ext in image_extensions:
        image_files.extend(image_dir.glob(f"*{ext}"))

    print(f"\n===== {split} =====")
    print(f"找到圖片：{len(image_files)}")

    for image_path in image_files:

        # ------------------------------
        # 讀取圖片
        # ------------------------------

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"無法讀取：{image_path}")
            continue

        height, width = image.shape[:2]

        # ------------------------------
        # 建立 Mask
        # ------------------------------

        mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        # ------------------------------
        # 對應 txt
        # ------------------------------

        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            print(f"找不到標註：{label_path}")
            continue

        # ------------------------------
        # 讀取 YOLO segmentation
        # ------------------------------

        with open(label_path, "r") as f:

            lines = f.readlines()

        for line in lines:

            values = line.strip().split()

            if len(values) < 7:
                continue

            # 第一個數字 = class
            class_id = int(values[0])

            if class_id not in CLASS_MAP:
                continue

            mask_value = CLASS_MAP[class_id]

            # ------------------------------
            # Polygon 座標
            # ------------------------------

            points = []

            coords = values[1:]

            for i in range(0, len(coords), 2):

                x = float(coords[i])
                y = float(coords[i + 1])

                # normalized → pixel
                px = int(x * width)
                py = int(y * height)

                points.append([px, py])

            points = np.array(
                points,
                dtype=np.int32
            )

            # ------------------------------
            # 填滿 Polygon
            # ------------------------------

            cv2.fillPoly(
                mask,
                [points],
                mask_value
            )

        # ------------------------------
        # 儲存 Mask
        # ------------------------------

        mask_path = output_mask_dir / f"{image_path.stem}.png"

        cv2.imwrite(
            str(mask_path),
            mask
        )

        # ------------------------------
        # 複製原始圖片
        # ------------------------------

        output_image_path = output_image_dir / image_path.name

        cv2.imwrite(
            str(output_image_path),
            image
        )

        print(f"完成：{image_path.name}")


# ==============================
# 執行
# ==============================

for split in ["train", "valid", "test"]:

    split_path = Path(SOURCE_DATASET) / split

    if split_path.exists():
        convert_split(split)

print("\n==============================")
print("YOLO → U-Net Mask 轉換完成")
print("==============================")