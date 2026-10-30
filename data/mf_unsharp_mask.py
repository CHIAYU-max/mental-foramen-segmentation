import os
import cv2
import numpy as np

def unsharp_mask(image, sigma=1.0, strength=1.5, threshold=0):
    """
    執行 Unsharp Mask (反銳化遮罩) 銳化處理
    :param image: 輸入圖片 (BGR)
    :param sigma: 高斯模糊半徑 (控制邊緣粗細)
    :param strength: 銳化強度 (越大邊緣越銳利)
    :param threshold: 閾值 (低於此差異的雜訊不放大，0 表示全部處理)
    """
    # 1. 產生高斯模糊影像
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    
    # 2. 計算銳化遮罩 (原圖 - 模糊圖)
    mask = cv2.subtract(image, blurred)
    
    # 3. 若有設定閾值，過濾微小雜訊
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        mask[low_contrast_mask] = 0

    # 4. 依照強度將遮罩疊加回原圖： sharpened = image + strength * mask
    sharpened = cv2.addWeighted(image, 1.0, mask, strength, 0)
    return sharpened

def process_folder(input_dir, output_dir, sigma=1.0, strength=1.5, threshold=0):
    """
    讀取資料夾內所有圖片，處理後輸出至指定目錄
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    
    # 建立輸出資料夾
    os.makedirs(output_dir, exist_ok=True)

    # 取得檔案清單
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    
    if not files:
        print(f"在 '{input_dir}' 中找不到支援的圖片檔案。")
        return

    print(f"找到 {len(files)} 張圖片，開始處理...")

    for idx, filename in enumerate(files, 1):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # 讀取圖片
        img = cv2.imread(input_path)
        if img is None:
            print(f"[{idx}/{len(files)}] 讀取失敗: {filename}")
            continue

        # 銳化處理
        result = unsharp_mask(img, sigma=sigma, strength=strength, threshold=threshold)

        # 儲存圖片
        cv2.imwrite(output_path, result)
        print(f"[{idx}/{len(files)}] 已處理: {filename}")

    print(f"\n所有圖片已成功輸出至: {output_dir}")

if __name__ == "__main__":
    # 設定輸入與輸出路徑
    INPUT_FOLDER = "dental_project/20261008/test"
    OUTPUT_FOLDER = "dental_project/20261008/um_test"

    # 參數調整：
    # sigma: 通常介於 0.5 ~ 2.0 (控制邊緣寬度)
    # strength: 通常介於 0.5 ~ 2.5 (銳化強度)
    # threshold: 0 ~ 10 (可避免放大暗處顆粒/雜訊)
    process_folder(
        input_dir=INPUT_FOLDER,
        output_dir=OUTPUT_FOLDER,
        sigma=1.0,
        strength=0.5,
        threshold=3
    )