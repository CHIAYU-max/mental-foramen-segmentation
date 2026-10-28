import cv2
import os
import numpy as np

def apply_weak_clahe(image, clip_limit=1.5, tile_size=(8, 8)):
    """
    套用微弱的 CLAHE 增強。
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img

def process_folder(input_folder, output_folder):
    # 1. 檢查輸入資料夾是否存在
    if not os.path.exists(input_folder):
        print(f"錯誤：找不到輸入資料夾 {input_folder}")
        return

    # 2. 自動建立輸出資料夾
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"已建立輸出目錄: {output_folder}")

    # 3. 定義支援的影像副檔名（忽略大小寫）
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

    # 4. 取得資料夾內所有檔案
    all_files = os.listdir(input_folder)
    image_files = [f for f in all_files if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"提示：在 {input_folder} 中沒有找到支援的影像檔案。")
        return

    print(f"開始處理，共找到 {len(image_files)} 張影像...")
    print("-" * 50)

    success_count = 0

    # 5. 開始批次處理
    for filename in image_files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # 支援中文/特殊路徑的讀取方式
        img = cv2.imdecode(np.fromfile(input_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"無法讀取檔案（可能毀損）: {filename}")
            continue

        # 執行 CLAHE 
        result = apply_weak_clahe(img, clip_limit=1.0, tile_size=(16, 16))

        # 支援中文/特殊路徑的寫入方式
        _, ext = os.path.splitext(output_path)
        success, encoded_img = cv2.imencode(ext, result)
        
        if success:
            encoded_img.tofile(output_path)
            print(f"成功處理: {filename} -> 已儲存")
            success_count += 1
        else:
            print(f"失敗：無法寫入檔案: {filename}")

    print("-" * 50)
    print(f"處理完成！成功：{success_count} / 總數：{len(image_files)}")

# --- 設定資料夾路徑 ---
# 這裡請填入你的「資料夾」路徑，而非單一檔案名稱
input_dir = "dental_project/20260913/mf_397_bf_gamma/images"   # 放原始照片的資料夾
output_dir = "dental_project/20260913/mf_397_bf_gamma_clahe/images"  # 增強後照片要存去的資料夾

process_folder(input_dir, output_dir)
