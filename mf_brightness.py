import os
import cv2
import numpy as np
import argparse
from pathlib import Path

def adjust_shadow_highlight_mask(image, shadow_boost=0.4, highlight_decay=0.4, radius=35):
    """
    使用高光/陰影遮罩演算法進行影像增強 (Photoshop 陰影/高光風格)。
    
    :param image: BGR 格式的 numpy 陣列
    :param shadow_boost: 陰影調亮強度 (0.0 ~ 1.0)
    :param highlight_decay: 亮部調暗強度 (0.0 ~ 1.0)
    :param radius: 遮罩模糊半徑 (用於局部調適，保持邊緣與細節對比)
    :return: 增強後的 BGR 影像
    """
    # 轉換至 LAB 顏色空間以處理 L (亮度) 通道
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # 歸一化到 [0, 1] 浮點數
    l_float = l_channel.astype(np.float32) / 255.0

    # 建立局部亮度基底 (Base Layer)
    if radius > 0:
        # 半徑確保為奇數
        ksize = radius if radius % 2 == 1 else radius + 1
        l_blur = cv2.GaussianBlur(l_float, (ksize, ksize), 0)
    else:
        l_blur = l_float

    # 1. 暗部調亮 (Shadow Lift Mask): 亮度越低，遮罩權重越高
    shadow_mask = 1.0 - l_blur
    shadow_gain = shadow_boost * np.power(shadow_mask, 2) * (1.0 - l_float)
    
    # 2. 亮部調暗 (Highlight Darken Mask): 亮度越高，遮罩權重越高
    highlight_mask = l_blur
    highlight_reduction = highlight_decay * np.power(highlight_mask, 2) * l_float

    # 套用調整
    l_enhanced = l_float + shadow_gain - highlight_reduction

    # 數值截斷在 [0, 1] 並轉回 8-bit
    l_enhanced = np.clip(l_enhanced * 255.0, 0, 255).astype(np.uint8)

    # 重組 LAB 空間並轉回 BGR
    enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr


def adjust_shadow_highlight_curve(image, strength=0.12):
    """
    使用 Smooth Sine Curve (正弦曲線) 進行全域動態範圍壓縮。
    公式: L_out = L + alpha * sin(2 * pi * L)
    - L < 0.5 (暗部) 時 sin > 0 -> 調亮暗部
    - L > 0.5 (亮部) 時 sin < 0 -> 調暗亮部
    
    :param image: BGR 格式的 numpy 陣列
    :param strength: 曲線強度 alpha (建議 0.05 ~ 0.15)
    :return: 增強後的 BGR 影像
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    l_float = l_channel.astype(np.float32) / 255.0
    
    # 計算正弦修正值
    delta = strength * np.sin(2.0 * np.pi * l_float)
    l_enhanced = l_float + delta
    
    l_enhanced = np.clip(l_enhanced * 255.0, 0, 255).astype(np.uint8)
    
    enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def read_image_unicode(file_path):
    """跨平台支援中文與特殊字元路徑的圖片讀取"""
    try:
        img_array = np.fromfile(str(file_path), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[錯誤] 讀取圖片失敗 {file_path}: {e}")
        return None


def write_image_unicode(file_path, image):
    """跨平台支援中文與特殊字元路徑的圖片寫入"""
    try:
        ext = Path(file_path).suffix
        success, encoded_img = cv2.imencode(ext, image)
        if success:
            encoded_img.tofile(str(file_path))
            return True
        return False
    except Exception as e:
        print(f"[錯誤] 寫入圖片失敗 {file_path}: {e}")
        return False


def process_folder(input_dir, output_dir, method="mask", shadow_boost=0.4, highlight_decay=0.4, radius=35, strength=0.12, recursive=False):
    """
    對整個資料夾內的影像進行「暗部調亮、亮部調暗」處理。
    
    :param input_dir: 輸入圖片資料夾路徑
    :param output_dir: 輸出圖片資料夾路徑
    :param method: 演算法選擇 ('mask' 或 'curve')
    :param shadow_boost: 陰影調亮強度 (mask 模式)
    :param highlight_decay: 亮部調暗強度 (mask 模式)
    :param radius: 局部遮罩半徑 (mask 模式)
    :param strength: 曲線強度 (curve 模式)
    :param recursive: 是否遞迴處理子資料夾
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"[錯誤] 輸入目錄不存在: {input_path.resolve()}")
        return

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    
    if recursive:
        all_files = [p for p in input_path.rglob('*') if p.suffix.lower() in valid_extensions]
    else:
        all_files = [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in valid_extensions]

    if not all_files:
        print(f"[提示] 在 {input_path} 中沒有找到支援的影像檔案。")
        return

    print(f"==================================================")
    print(f" 影像亮部調暗 / 暗部調亮 批次處理")
    print(f" 演算法: {method.upper()}")
    print(f" 總檔案數量: {len(all_files)}")
    print(f" 輸入目錄: {input_path.resolve()}")
    print(f" 輸出目錄: {output_path.resolve()}")
    print(f"==================================================")

    success_count = 0
    fail_count = 0

    for idx, img_file in enumerate(all_files, 1):
        # 計算相對路徑以維持子資料夾結構
        rel_path = img_file.relative_to(input_path)
        save_file = output_path / rel_path

        # 建立輸出對應的子資料夾
        save_file.parent.mkdir(parents=True, exist_ok=True)

        # 讀取影像
        img = read_image_unicode(img_file)
        if img is None:
            fail_count += 1
            continue

        # 執行演算法
        if method == "curve":
            result = adjust_shadow_highlight_curve(img, strength=strength)
        else:
            result = adjust_shadow_highlight_mask(
                img, 
                shadow_boost=shadow_boost, 
                highlight_decay=highlight_decay, 
                radius=radius
            )

        # 寫入影像
        if write_image_unicode(save_file, result):
            success_count += 1
            print(f"[{idx}/{len(all_files)}] 已處理: {rel_path}")
        else:
            fail_count += 1
            print(f"[{idx}/{len(all_files)}] 失敗(無法儲存): {rel_path}")

    print(f"==================================================")
    print(f"處理完成！ 成功: {success_count} | 失敗: {fail_count}")
    print(f"==================================================")


if __name__ == "__main__":
    # =========================================================================
    #  請在此處直接修改「輸入資料夾」與「輸出資料夾」路徑：
    # =========================================================================
    INPUT_DIR = "dental_project/20260919/test_um/images"   # 原始圖片資料夾路徑
    OUTPUT_DIR = "dental_project/20260919/test_um_brightness/images"     # 增強後圖片儲存路徑

    # --- 增強參數設定 ---
    METHOD = "mask"          # 增強模式：'mask' (遮罩局部調態，推薦) 或 'curve' (全域正弦曲線)
    SHADOW_BOOST = 0.2       # 暗部調亮強度 (0.0 ~ 1.0)
    HIGHLIGHT_DECAY = 0.2    # 亮部調暗強度 (0.0 ~ 1.0)
    RADIUS = 35              # 局部遮罩半徑 (建議 15~51 的奇數)
    STRENGTH = 0.12          # [Curve模式] S曲線調節強度 (0.05~0.15)
    RECURSIVE = False        # 是否遞迴處理子資料夾內的圖片
    # =========================================================================

    # 執行批次處理
    process_folder(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        method=METHOD,
        shadow_boost=SHADOW_BOOST,
        highlight_decay=HIGHLIGHT_DECAY,
        radius=RADIUS,
        strength=STRENGTH,
        recursive=RECURSIVE
    )

