import cv2
import os

def enhance_dental_xrays(input_folder, output_folder):
    """
    讀取資料夾中的牙齒環口片，使用 CLAHE 增強牙齒輪廓，並輸出到新資料夾。
    """
    # 檢查並建立輸出資料夾
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"已建立輸出資料夾: {output_folder}")

    # 定義支援的圖片副檔名
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    
    # 記錄處理數量
    processed_count = 0

    for filename in os.listdir(input_folder):
        # 確保只處理圖片檔案
        if filename.lower().endswith(valid_extensions):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 1. 讀取圖片 (強制以灰階模式讀取，X 光片通常是單通道灰階的)
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"警告: 無法讀取圖片 {filename}，已略過。")
                continue

            # (可選步驟) 雙邊濾波器降噪
            # 如果原圖雜訊較多，可以在 CLAHE 前先做雙邊濾波。它能保留邊緣(牙齒輪廓)同時消除平滑區域雜訊。
            # img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

            # 2. 建立 CLAHE 物件
            # clipLimit: 限制對比度的閾值。數值越大對比越強，但雜訊也會變多。醫療影像建議在 2.0 ~ 4.0 之間微調。
            # tileGridSize: 網格大小。將影像切分成 8x8 的區塊進行局部均衡化。
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

            # 3. 應用 CLAHE 進行影像增強
            enhanced_img = clahe.apply(img)

            # 4. 儲存增強後的圖片
            cv2.imwrite(output_path, enhanced_img)
            print(f"已處理並儲存: {filename}")
            processed_count += 1

    print(f"\n處理完成！共處理了 {processed_count} 張圖片。")

# ==========================================
# 執行設定區塊
# ==========================================
if __name__ == "__main__":
    # 請將以下路徑替換成你實際的資料夾名稱或路徑
    # 可以是相對路徑 (例如相對於此 Python 腳本的位置) 或絕對路徑
    INPUT_DIR = "original_dataset/images"   # 原始環口片資料夾
    OUTPUT_DIR = "original_dataset/enhanced_images" # 增強後輸出的資料夾

    # 如果輸入資料夾不存在，給予提示
    if not os.path.exists(INPUT_DIR):
        print(f"錯誤: 找不到輸入資料夾 '{INPUT_DIR}'，請確認路徑是否正確。")
    else:
        enhance_dental_xrays(INPUT_DIR, OUTPUT_DIR)