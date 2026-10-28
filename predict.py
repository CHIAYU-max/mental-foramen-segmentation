import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn


# ============================================================
# 1. 路徑設定
# ============================================================

# ------------------------------------------------------------
# Best U-Net 模型
# ------------------------------------------------------------

MODEL_PATH = r".\best_unet.pth"


# ------------------------------------------------------------
# Test X-ray 圖片
# ------------------------------------------------------------

TEST_DIR = r".\datasets\mf_ian_unet\test\images"


# ------------------------------------------------------------
# Test Ground Truth Mask
#
# 每一張 Test 圖都要有對應的 Ground Truth Mask
#
# 例如：
#
# images/
#   001.png
#   002.png
#
# masks/
#   001.png
#   002.png
#
# ------------------------------------------------------------

GT_MASK_DIR = r".\datasets\mf_ian_unet\test\masks"


# ------------------------------------------------------------
# 預測結果
# ------------------------------------------------------------

OUTPUT_DIR = r".\test_predictions"


# ============================================================
# 2. 模型設定
# ============================================================

IMAGE_SIZE = 512

NUM_CLASSES = 3

# Class 定義
#
# 0 = Background
# 1 = MF 頦孔
# 2 = IAN 下齒槽神經


# ============================================================
# 3. 雙邊濾波設定
# ============================================================

# 注意：
# 這裡一定要和你「訓練 U-Net 時使用的雙邊濾波參數」
# 完全一致。
#
# 如果你的訓練程式使用不同參數，
# 請把下面三個數字改成訓練時的設定。

BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75


# ============================================================
# 4. Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 5. U-Net DoubleConv
# ============================================================

class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        # 必須使用 block
        # 因為你的 best_unet.pth 使用的是：
        #
        # enc1.block
        # enc2.block
        # enc3.block
        # enc4.block
        # bottleneck.block
        # dec4.block
        # dec3.block
        # dec2.block
        # dec1.block

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )


    def forward(self, x):

        return self.block(x)


# ============================================================
# 6. U-Net
# ============================================================

class UNet(nn.Module):

    def __init__(
        self,
        num_classes=3
    ):

        super().__init__()


        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        self.enc1 = DoubleConv(
            1,
            64
        )

        self.enc2 = DoubleConv(
            64,
            128
        )

        self.enc3 = DoubleConv(
            128,
            256
        )

        self.enc4 = DoubleConv(
            256,
            512
        )


        # ----------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------

        self.bottleneck = DoubleConv(
            512,
            1024
        )


        # ----------------------------------------------------
        # Pooling
        # ----------------------------------------------------

        self.pool = nn.MaxPool2d(
            kernel_size=2
        )


        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.up4 = nn.ConvTranspose2d(
            1024,
            512,
            kernel_size=2,
            stride=2
        )

        self.dec4 = DoubleConv(
            1024,
            512
        )


        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            512,
            256
        )


        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            256,
            128
        )


        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            128,
            64
        )


        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        self.out_conv = nn.Conv2d(
            64,
            num_classes,
            kernel_size=1
        )


    def forward(self, x):

        # ====================================================
        # Encoder
        # ====================================================

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )


        # ====================================================
        # Bottleneck
        # ====================================================

        b = self.bottleneck(
            self.pool(e4)
        )


        # ====================================================
        # Decoder
        # ====================================================

        d4 = self.up4(b)

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = self.dec4(d4)


        d3 = self.up3(d4)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)


        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)


        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)


        # ====================================================
        # Output
        # ====================================================

        output = self.out_conv(d1)

        return output


# ============================================================
# 7. 建立輸出資料夾
# ============================================================

OUTPUT_DIR = Path(
    OUTPUT_DIR
)

MASK_DIR = OUTPUT_DIR / "masks"

OVERLAY_DIR = OUTPUT_DIR / "overlays"

MF_DIR = OUTPUT_DIR / "mf_masks"

IAN_DIR = OUTPUT_DIR / "ian_masks"

RESULT_DIR = OUTPUT_DIR / "evaluation"


MASK_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OVERLAY_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MF_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IAN_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 8. Metric Functions
# ============================================================

def calculate_metrics(
    prediction,
    ground_truth
):
    """
    計算二元 segmentation metrics

    prediction:
        0 / 1 的 binary mask

    ground_truth:
        0 / 1 的 binary mask

    回傳：
        Dice
        IoU
        Precision
        Recall
        F1
    """

    prediction = prediction.astype(bool)

    ground_truth = ground_truth.astype(bool)


    # --------------------------------------------------------
    # True Positive
    # --------------------------------------------------------

    tp = np.logical_and(
        prediction,
        ground_truth
    ).sum()


    # --------------------------------------------------------
    # False Positive
    # --------------------------------------------------------

    fp = np.logical_and(
        prediction,
        np.logical_not(ground_truth)
    ).sum()


    # --------------------------------------------------------
    # False Negative
    # --------------------------------------------------------

    fn = np.logical_and(
        np.logical_not(prediction),
        ground_truth
    ).sum()


    # --------------------------------------------------------
    # Dice
    # --------------------------------------------------------

    dice_denominator = (
        2 * tp + fp + fn
    )

    if dice_denominator == 0:

        dice = 1.0

    else:

        dice = (
            2 * tp
        ) / dice_denominator


    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    iou_denominator = (
        tp + fp + fn
    )

    if iou_denominator == 0:

        iou = 1.0

    else:

        iou = (
            tp
        ) / iou_denominator


    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision_denominator = (
        tp + fp
    )

    if precision_denominator == 0:

        precision = 1.0

    else:

        precision = (
            tp
        ) / precision_denominator


    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall_denominator = (
        tp + fn
    )

    if recall_denominator == 0:

        recall = 1.0

    else:

        recall = (
            tp
        ) / recall_denominator


    # --------------------------------------------------------
    # F1-score
    #
    # Binary segmentation 裡 F1 和 Dice 數學上相同
    # --------------------------------------------------------

    f1_denominator = (
        precision + recall
    )

    if f1_denominator == 0:

        f1 = 0.0

    else:

        f1 = (
            2 * precision * recall
        ) / f1_denominator


    return (
        dice,
        iou,
        precision,
        recall,
        f1
    )


# ============================================================
# 9. 顯示基本資訊
# ============================================================

print()

print("=" * 80)

print(
    "U-Net TEST PREDICTION + EVALUATION"
)

print("=" * 80)

print()

print(
    "Device:",
    DEVICE
)

print(
    "Model:",
    MODEL_PATH
)

print(
    "Test:",
    TEST_DIR
)

print(
    "Ground Truth:",
    GT_MASK_DIR
)

print(
    "Output:",
    OUTPUT_DIR
)

print()

print(
    "Image Size:",
    IMAGE_SIZE
)

print(
    "Bilateral Filter:",
    BILATERAL_D,
    BILATERAL_SIGMA_COLOR,
    BILATERAL_SIGMA_SPACE
)

print()


# ============================================================
# 10. 檢查模型
# ============================================================

if not os.path.exists(
    MODEL_PATH
):

    raise FileNotFoundError(
        f"找不到模型：{MODEL_PATH}"
    )


# ============================================================
# 11. 檢查 Test
# ============================================================

test_path = Path(
    TEST_DIR
)

if not test_path.exists():

    raise FileNotFoundError(
        f"找不到 Test 資料夾：{TEST_DIR}"
    )


# ============================================================
# 12. 檢查 Ground Truth
# ============================================================

gt_path = Path(
    GT_MASK_DIR
)

if not gt_path.exists():

    raise FileNotFoundError(
        f"找不到 Ground Truth Mask 資料夾：{GT_MASK_DIR}"
    )


# ============================================================
# 13. 建立 U-Net
# ============================================================

model = UNet(
    num_classes=NUM_CLASSES
)

model = model.to(
    DEVICE
)


# ============================================================
# 14. 載入 Best Model
# ============================================================

print(
    "Loading best model..."
)

print()


checkpoint = torch.load(

    MODEL_PATH,

    map_location=DEVICE,

    weights_only=False

)


# ============================================================
# 15. 取得 model_state_dict
# ============================================================

if isinstance(
    checkpoint,
    dict
):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


# ============================================================
# 16. 載入權重
# ============================================================

model.load_state_dict(
    state_dict
)

model.eval()


print(
    "Best model loaded successfully."
)

print()


# ============================================================
# 17. Checkpoint 資訊
# ============================================================

if isinstance(
    checkpoint,
    dict
):

    if "epoch" in checkpoint:

        print(
            "Best model epoch:",
            checkpoint["epoch"]
        )


    if "mean_dice" in checkpoint:

        print(
            "Best Mean Dice:",
            checkpoint["mean_dice"]
        )


print()


# ============================================================
# 18. 找 Test 圖片
# ============================================================

extensions = [
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.bmp",
    "*.tif",
    "*.tiff"
]


image_files = []


for extension in extensions:

    image_files.extend(
        test_path.glob(extension)
    )


image_files = sorted(
    image_files
)


print(
    "Test images:",
    len(image_files)
)

print()


if len(image_files) == 0:

    raise RuntimeError(
        "Test 資料夾裡沒有找到圖片。"
    )


# ============================================================
# 19. 儲存每張圖片結果
# ============================================================

results = []


# ============================================================
# 20. 開始 Prediction + Evaluation
# ============================================================

print("=" * 80)

print(
    "START PREDICTION + EVALUATION"
)

print("=" * 80)

print()


with torch.no_grad():

    for index, image_path in enumerate(
        image_files
    ):

        print(
            f"[{index + 1}/{len(image_files)}]"
        )

        print(
            "Image:",
            image_path.name
        )


        # ====================================================
        # 讀取 X-ray
        # ====================================================

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )


        if image is None:

            print(
                "ERROR: 無法讀取圖片"
            )

            print()

            continue


        original_h, original_w = image.shape


        # ====================================================
        # 雙邊濾波
        # ====================================================

        filtered_image = cv2.bilateralFilter(

            image,

            BILATERAL_D,

            BILATERAL_SIGMA_COLOR,

            BILATERAL_SIGMA_SPACE

        )


        # ====================================================
        # Resize
        # ====================================================

        resized_image = cv2.resize(

            filtered_image,

            (
                IMAGE_SIZE,
                IMAGE_SIZE
            ),

            interpolation=cv2.INTER_AREA

        )


        # ====================================================
        # Normalize
        # ====================================================

        normalized_image = (
            resized_image.astype(
                np.float32
            ) / 255.0
        )


        # ====================================================
        # Tensor
        # ====================================================

        input_tensor = torch.from_numpy(
            normalized_image
        )


        # [H,W]
        # ↓
        # [1,H,W]
        # ↓
        # [1,1,H,W]

        input_tensor = (
            input_tensor
            .unsqueeze(0)
            .unsqueeze(0)
        )


        input_tensor = input_tensor.to(
            DEVICE
        )


        # ====================================================
        # U-Net Prediction
        # ====================================================

        output = model(
            input_tensor
        )


        # ====================================================
        # Argmax
        # ====================================================

        prediction = torch.argmax(
            output,
            dim=1
        )


        prediction = prediction.squeeze(
            0
        )


        prediction = prediction.cpu().numpy().astype(
            np.uint8
        )


        # ====================================================
        # Resize Prediction 回原始大小
        # ====================================================

        prediction = cv2.resize(

            prediction,

            (
                original_w,
                original_h
            ),

            interpolation=cv2.INTER_NEAREST

        )


        # ====================================================
        # Class Mask
        # ====================================================

        mf_mask = (
            prediction == 1
        )

        ian_mask = (
            prediction == 2
        )


        # ====================================================
        # 讀取 Ground Truth Mask
        # ====================================================

        gt_file = None


        # 優先尋找相同檔名
        candidate = (
            gt_path /
            image_path.name
        )


        if candidate.exists():

            gt_file = candidate

        else:

            # 如果副檔名不同，
            # 依照 stem 尋找

            possible_files = []

            for ext in extensions:

                possible_files.extend(
                    gt_path.glob(
                        image_path.stem + ext.replace("*", "")
                    )
                )

            if len(possible_files) > 0:

                gt_file = possible_files[0]


        if gt_file is None:

            print(
                "WARNING: 找不到 Ground Truth Mask：",
                image_path.stem
            )

            print()

            continue


        # ====================================================
        # 讀取 Ground Truth
        # ====================================================

        gt_mask = cv2.imread(

            str(gt_file),

            cv2.IMREAD_GRAYSCALE

        )


        if gt_mask is None:

            print(
                "WARNING: 無法讀取 Ground Truth：",
                gt_file
            )

            print()

            continue


        # ====================================================
        # 如果 GT 尺寸不同，Resize
        # ====================================================

        if gt_mask.shape != (
            original_h,
            original_w
        ):

            gt_mask = cv2.resize(

                gt_mask,

                (
                    original_w,
                    original_h
                ),

                interpolation=cv2.INTER_NEAREST

            )


        # ====================================================
        # 顯示 GT 中有哪些值
        # ====================================================

        unique_values = np.unique(
            gt_mask
        )


        # ====================================================
        # Ground Truth Class
        #
        # 這裡支援兩種格式：
        #
        # 格式 A：
        # 0 = Background
        # 1 = MF
        # 2 = IAN
        #
        # 格式 B：
        # 0 = Background
        # 128 = MF
        # 255 = IAN
        #
        # ====================================================

        if (
            1 in unique_values
            or 2 in unique_values
        ):

            gt_mf_mask = (
                gt_mask == 1
            )

            gt_ian_mask = (
                gt_mask == 2
            )

        elif (
            128 in unique_values
            or 255 in unique_values
        ):

            gt_mf_mask = (
                gt_mask == 128
            )

            gt_ian_mask = (
                gt_mask == 255
            )

        else:

            print(
                "WARNING: Ground Truth Mask 的類別值不符合預期：",
                unique_values
            )

            print()

            continue


        # ====================================================
        # MF Metrics
        # ====================================================

        (
            mf_dice,
            mf_iou,
            mf_precision,
            mf_recall,
            mf_f1
        ) = calculate_metrics(

            mf_mask,

            gt_mf_mask

        )


        # ====================================================
        # IAN Metrics
        # ====================================================

        (
            ian_dice,
            ian_iou,
            ian_precision,
            ian_recall,
            ian_f1
        ) = calculate_metrics(

            ian_mask,

            gt_ian_mask

        )


        # ====================================================
        # Mean Metrics
        # ====================================================

        mean_dice = (
            mf_dice +
            ian_dice
        ) / 2.0


        mean_iou = (
            mf_iou +
            ian_iou
        ) / 2.0


        mean_precision = (
            mf_precision +
            ian_precision
        ) / 2.0


        mean_recall = (
            mf_recall +
            ian_recall
        ) / 2.0


        mean_f1 = (
            mf_f1 +
            ian_f1
        ) / 2.0


        # ====================================================
        # 儲存 Mask
        # ====================================================

        mask = np.zeros(

            (
                original_h,
                original_w
            ),

            dtype=np.uint8

        )


        mask[mf_mask] = 128

        mask[ian_mask] = 255


        mask_path = (
            MASK_DIR /
            f"{image_path.stem}_mask.png"
        )


        cv2.imwrite(

            str(mask_path),

            mask

        )


        # ====================================================
        # MF Mask
        # ====================================================

        mf_only = np.zeros(

            (
                original_h,
                original_w
            ),

            dtype=np.uint8

        )


        mf_only[mf_mask] = 255


        mf_path = (
            MF_DIR /
            f"{image_path.stem}_MF.png"
        )


        cv2.imwrite(

            str(mf_path),

            mf_only

        )


        # ====================================================
        # IAN Mask
        # ====================================================

        ian_only = np.zeros(

            (
                original_h,
                original_w
            ),

            dtype=np.uint8

        )


        ian_only[ian_mask] = 255


        ian_path = (
            IAN_DIR /
            f"{image_path.stem}_IAN.png"
        )


        cv2.imwrite(

            str(ian_path),

            ian_only

        )


        # ====================================================
        # Overlay
        # ====================================================

        overlay = cv2.cvtColor(

            image,

            cv2.COLOR_GRAY2BGR

        )


        color_layer = overlay.copy()


        # ----------------------------------------------------
        # MF → 紅色
        # ----------------------------------------------------

        color_layer[mf_mask] = (
            0,
            0,
            255
        )


        # ----------------------------------------------------
        # IAN → 綠色
        # ----------------------------------------------------

        color_layer[ian_mask] = (
            0,
            255,
            0
        )


        overlay = cv2.addWeighted(

            overlay,
            0.60,

            color_layer,
            0.40,

            0

        )


        overlay_path = (
            OVERLAY_DIR /
            f"{image_path.stem}_overlay.png"
        )


        cv2.imwrite(

            str(overlay_path),

            overlay

        )


        # ====================================================
        # 儲存結果
        # ====================================================

        results.append({

            "Image": image_path.name,

            "MF_Dice": mf_dice,

            "MF_IoU": mf_iou,

            "MF_Precision": mf_precision,

            "MF_Recall": mf_recall,

            "MF_F1": mf_f1,

            "IAN_Dice": ian_dice,

            "IAN_IoU": ian_iou,

            "IAN_Precision": ian_precision,

            "IAN_Recall": ian_recall,

            "IAN_F1": ian_f1,

            "Mean_Dice": mean_dice,

            "Mean_IoU": mean_iou,

            "Mean_Precision": mean_precision,

            "Mean_Recall": mean_recall,

            "Mean_F1": mean_f1

        })


        # ====================================================
        # 顯示結果
        # ====================================================

        print(
            f"  MF Dice      : {mf_dice:.4f}"
        )

        print(
            f"  MF IoU       : {mf_iou:.4f}"
        )

        print(
            f"  MF Precision : {mf_precision:.4f}"
        )

        print(
            f"  MF Recall    : {mf_recall:.4f}"
        )

        print(
            f"  MF F1        : {mf_f1:.4f}"
        )

        print()

        print(
            f"  IAN Dice      : {ian_dice:.4f}"
        )

        print(
            f"  IAN IoU       : {ian_iou:.4f}"
        )

        print(
            f"  IAN Precision : {ian_precision:.4f}"
        )

        print(
            f"  IAN Recall    : {ian_recall:.4f}"
        )

        print(
            f"  IAN F1        : {ian_f1:.4f}"
        )

        print()

        print(
            f"  Mean Dice     : {mean_dice:.4f}"
        )

        print(
            f"  Mean IoU      : {mean_iou:.4f}"
        )

        print(
            f"  Mean Precision: {mean_precision:.4f}"
        )

        print(
            f"  Mean Recall   : {mean_recall:.4f}"
        )

        print(
            f"  Mean F1       : {mean_f1:.4f}"
        )

        print()

        print("-" * 80)

        print()


# ============================================================
# 21. 檢查是否有成功評估
# ============================================================

if len(results) == 0:

    raise RuntimeError(
        "沒有任何圖片成功完成 Ground Truth 評估。"
    )


# ============================================================
# 22. 建立 DataFrame
# ============================================================

df = pd.DataFrame(
    results
)


# ============================================================
# 23. 計算 Test 平均
# ============================================================

average_result = {

    "Image": "AVERAGE",

    "MF_Dice":
        df["MF_Dice"].mean(),

    "MF_IoU":
        df["MF_IoU"].mean(),

    "MF_Precision":
        df["MF_Precision"].mean(),

    "MF_Recall":
        df["MF_Recall"].mean(),

    "MF_F1":
        df["MF_F1"].mean(),

    "IAN_Dice":
        df["IAN_Dice"].mean(),

    "IAN_IoU":
        df["IAN_IoU"].mean(),

    "IAN_Precision":
        df["IAN_Precision"].mean(),

    "IAN_Recall":
        df["IAN_Recall"].mean(),

    "IAN_F1":
        df["IAN_F1"].mean(),

    "Mean_Dice":
        df["Mean_Dice"].mean(),

    "Mean_IoU":
        df["Mean_IoU"].mean(),

    "Mean_Precision":
        df["Mean_Precision"].mean(),

    "Mean_Recall":
        df["Mean_Recall"].mean(),

    "Mean_F1":
        df["Mean_F1"].mean()

}


# ============================================================
# 24. 加入平均結果
# ============================================================

df = pd.concat(

    [
        df,
        pd.DataFrame(
            [average_result]
        )
    ],

    ignore_index=True

)


# ============================================================
# 25. 輸出 CSV
# ============================================================

csv_path = (
    RESULT_DIR /
    "test_metrics.csv"
)


df.to_csv(

    csv_path,

    index=False,

    encoding="utf-8-sig"

)


# ============================================================
# 26. 顯示最後結果
# ============================================================

print()

print("=" * 80)

print(
    "TEST EVALUATION RESULT"
)

print("=" * 80)

print()


print(
    "Test images evaluated:",
    len(results)
)

print()


print("----------- MF -----------")

print(
    f"Dice      : {average_result['MF_Dice']:.4f}"
)

print(
    f"IoU       : {average_result['MF_IoU']:.4f}"
)

print(
    f"Precision : {average_result['MF_Precision']:.4f}"
)

print(
    f"Recall    : {average_result['MF_Recall']:.4f}"
)

print(
    f"F1-score  : {average_result['MF_F1']:.4f}"
)

print()


print("----------- IAN -----------")

print(
    f"Dice      : {average_result['IAN_Dice']:.4f}"
)

print(
    f"IoU       : {average_result['IAN_IoU']:.4f}"
)

print(
    f"Precision : {average_result['IAN_Precision']:.4f}"
)

print(
    f"Recall    : {average_result['IAN_Recall']:.4f}"
)

print(
    f"F1-score  : {average_result['IAN_F1']:.4f}"
)

print()


print("----------- Mean -----------")

print(
    f"Mean Dice      : {average_result['Mean_Dice']:.4f}"
)

print(
    f"Mean IoU       : {average_result['Mean_IoU']:.4f}"
)

print(
    f"Mean Precision : {average_result['Mean_Precision']:.4f}"
)

print(
    f"Mean Recall    : {average_result['Mean_Recall']:.4f}"
)

print(
    f"Mean F1        : {average_result['Mean_F1']:.4f}"
)

print()


print(
    "CSV saved to:"
)

print(
    csv_path
)

print()


print(
    "Prediction masks:"
)

print(
    MASK_DIR
)

print()


print(
    "Overlay:"
)

print(
    OVERLAY_DIR
)

print()


print("=" * 80)

print(
    "ALL DONE"
)

print("=" * 80)