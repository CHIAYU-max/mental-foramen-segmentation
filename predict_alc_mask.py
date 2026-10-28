from ultralytics import YOLO
import os
import cv2
import glob
import json
import numpy as np

def crop_lower_mandible_and_save(model_path, source_dir, output_crop_dir, output_coord_dir, conf_thres=0.25):
    """
    載入牙齒模型，利用「咬合線」精準裁切下顎區域，並將最終結果「左右對半切」。
    包含：
    1. 防呆機制：處理嚴重缺牙或無牙的極端案例。
    2. 左側翻轉：將左側影像水平翻轉，統一特徵方向。
    3. 動態比例延伸：取代寫死的像素值，適應不同解析度的影像。
    """
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型檔案: {model_path}")
        return

    os.makedirs(output_crop_dir, exist_ok=True)
    os.makedirs(output_coord_dir, exist_ok=True)

    print(f"⚖️ 正在載入模型: {model_path} ...")
    model = YOLO(model_path)

    img_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        img_paths.extend(glob.glob(os.path.join(source_dir, ext)))

    if not img_paths:
        print(f"⚠️ 找不到任何圖片！請檢查路徑: {source_dir}")
        return

    print(f"🚀 開始預測並裁切 {len(img_paths)} 張圖片...")

    for img_path in img_paths:
        file_name = os.path.basename(img_path)
        base_name = os.path.splitext(file_name)[0]

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ 無法讀取圖片: {file_name}")
            continue
        h, w = img.shape[:2]

        # ================= ✂️ 動態裁切範圍設定 =================
        # 捨棄寫死的像素，改用圖片寬高的比例，適應不同解析度 (如 4K 或壓縮圖)
        MARGIN_X = int(w * 0.02)          # 左右稍微延伸 2% 寬度
        MARGIN_BITE_LINE = int(h * 0.01)  # 往上留 1% 高度當作緩衝
        EXTEND_Y_BOTTOM = int(h * 0.25)   # 🌟 從下排最底部往下延伸 25% 高度
        # ===================================================

        results = model.predict(img_path, conf=conf_thres, verbose=False)
        result = results[0]

        # ================= 🛡️ 核心裁切與防呆邏輯 =================
        if len(result.boxes) == 0:
            print(f"⚠️ {file_name} 完全沒有偵測到牙齒！啟動備用裁切機制。")
            crop_x_min, crop_x_max = 0, w
            crop_y_min, crop_y_max = int(h * 0.55), h
            
        else:
            boxes = result.boxes.xyxy.cpu().numpy()
            
            global_x_min = int(boxes[:, 0].min())
            global_x_max = int(boxes[:, 2].max())
            crop_x_min = max(0, global_x_min - MARGIN_X)
            crop_x_max = min(w, global_x_max + MARGIN_X)
            
            y_centers = (boxes[:, 1] + boxes[:, 3]) / 2.0
            bite_line_y = np.median(y_centers)
            
            lower_teeth_boxes = boxes[y_centers > bite_line_y]
            
            if len(lower_teeth_boxes) < 3:
                print(f"⚠️ {file_name} 下排牙齒缺失！啟動備用裁切機制。")
                crop_y_min, crop_y_max = int(h * 0.55), h
            else:
                lower_y_max = int(lower_teeth_boxes[:, 3].max())
                crop_y_min = max(0, int(bite_line_y) - MARGIN_BITE_LINE) 
                crop_y_max = min(h, lower_y_max + EXTEND_Y_BOTTOM)

        # ================= 🪚 左右對半切與翻轉邏輯 =================
        crop_x_mid = (crop_x_min + crop_x_max) // 2

        # 裁切影像
        cropped_left = img[crop_y_min:crop_y_max, crop_x_min:crop_x_mid]
        cropped_right = img[crop_y_min:crop_y_max, crop_x_mid:crop_x_max]

        # 🌟 執行水平翻轉 (讓左側骨骼走勢與右側一致)
        cropped_left_flipped = cv2.flip(cropped_left, 1)

        cv2.imwrite(os.path.join(output_crop_dir, f"{base_name}_L.jpg"), cropped_left_flipped)
        cv2.imwrite(os.path.join(output_crop_dir, f"{base_name}_R.jpg"), cropped_right)

        # ================= 📝 儲存座標資訊到 JSON =================
        # 必須在 JSON 記錄是否翻轉，以便未來將預測座標還原回原圖
        left_coord_info = {
            "image_name": file_name,
            "original_width": w,
            "original_height": h,
            "crop_x_min": int(crop_x_min),
            "crop_y_min": int(crop_y_min),
            "crop_width": int(crop_x_mid - crop_x_min),
            "crop_height": int(crop_y_max - crop_y_min),
            "side": "Left",
            "is_flipped": True  # 🌟 新增標記
        }
        with open(os.path.join(output_coord_dir, f"{base_name}_L_info.json"), 'w', encoding='utf-8') as f:
            json.dump(left_coord_info, f, indent=4)

        right_coord_info = {
            "image_name": file_name,
            "original_width": w,
            "original_height": h,
            "crop_x_min": int(crop_x_mid),
            "crop_y_min": int(crop_y_min),
            "crop_width": int(crop_x_max - crop_x_mid),
            "crop_height": int(crop_y_max - crop_y_min),
            "side": "Right",
            "is_flipped": False # 🌟 新增標記
        }
        with open(os.path.join(output_coord_dir, f"{base_name}_R_info.json"), 'w', encoding='utf-8') as f:
            json.dump(right_coord_info, f, indent=4)

    print(f"\n🎉 處理完成！")

if __name__ == "__main__":
    my_model_path = r"./dental_project/train_success_replica3/weights/best.pt"
    test_images_dir = r"./datasets/20251217_alc_66/train/images"
    output_cropped_folder = r"./dental_project/20260408_mf_131/cropped_images"
    output_coords_folder = r"./dental_project/20260408_mf_131/crop_info"
    
    crop_lower_mandible_and_save(my_model_path, test_images_dir, output_cropped_folder, output_coords_folder, 0.25)