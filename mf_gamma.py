import os
import cv2
import numpy as np

def adaptive_gamma_correction(image):
    """
    對輸入影像進行自適應 Gamma 亮度增強。
    保持色相(H)與飽和度(S)不變，僅調整亮度(V)。
    """
    # 轉換至 HSV 空間
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # 正規化亮度值至 [0, 1]
    v_norm = v / 255.0
    
    # 計算亮度均值
    mean_val = np.mean(v_norm)
    
    # 根據平均亮度計算自適應 gamma
    # 當圖片偏暗 (mean_val < 0.5) 時，gamma < 1 提亮暗部
    # 當圖片偏亮 (mean_val > 0.5) 時，gamma >= 1 抑制過曝
    gamma = -np.log2(max(mean_val, 1e-5))
    gamma = np.clip(gamma, 0.4, 2.5)  # 限制 gamma 範圍避免極端失真
    
    # 自適應 Gamma 變換
    v_enhanced = np.power(v_norm, gamma)
    v_enhanced = np.uint8(v_enhanced * 255)
    
    # 合併頻道並轉回 BGR
    enhanced_hsv = cv2.merge([h, s, v_enhanced])
    enhanced_bgr = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
    
    return enhanced_bgr, gamma

def batch_process(input_dir, output_dir):
    """
    批次處理指定資料夾中的所有圖片。
    """
    os.makedirs(output_dir, exist_ok=True)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    
    file_list = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    
    if not file_list:
        print(f"在 '{input_dir}' 中找不到支援的圖片檔案。")
        return

    print(f"開始處理，共找到 {len(file_list)} 張圖片...")
    
    for filename in file_list:
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"無法讀取圖片：{filename}")
            continue
            
        enhanced_img, gamma_val = adaptive_gamma_correction(img)
        
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, enhanced_img)
        print(f"已完成：{filename} (自適應 γ = {gamma_val:.2f})")
        
    print(f"所有圖片處理完成，儲存於：{output_dir}")

if __name__ == "__main__":
    # 設定輸入與輸出資料夾路徑
    INPUT_FOLDER = "dental_project/20260913/mf_559_bilateral_filter/images"
    OUTPUT_FOLDER = "dental_project/20260913/mf_397_bf_gamma/images"
    
    # 建立範例輸入資料夾（若不存在）
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    
    batch_process(INPUT_FOLDER, OUTPUT_FOLDER)