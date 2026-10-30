import json
import os
from pathlib import Path
from PIL import Image

def convert_labelme_to_yolov8_seg(json_dir, output_dir, class_mapping):
    """
    將 Labelme JSON 標註檔轉換為 YOLOv8-seg 格式的 txt 檔
    
    :param json_dir: 存放 JSON 檔案的資料夾路徑
    :param output_dir: 轉換後 TXT 檔案的輸出資料夾路徑
    :param class_mapping: 類別名稱對應 ID 的字典，例如 {"cat": 0, "dog": 1}
    """
    json_dir = Path(json_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(json_dir.glob("*.json"))
    if not json_files:
        print(f"在 {json_dir} 中沒有找到任何 .json 檔案。")
        return

    converted_count = 0

    for json_path in json_files:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 取得影像尺寸
        img_w = data.get("imageWidth")
        img_h = data.get("imageHeight")

        # 若 JSON 內沒有尺寸資訊，嘗試自動讀取同名的圖片
        if not img_w or not img_h:
            img_name = data.get("imagePath")
            img_path = json_dir / img_name if img_name else None
            if img_path and img_path.exists():
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
            else:
                print(f"[Warning] 跳過 {json_path.name}: 無法取得圖片寬高資訊。")
                continue

        yolo_lines = []

        for shape in data.get("shapes", []):
            label = shape.get("label")
            shape_type = shape.get("shape_type", "polygon")

            # 僅處理 polygon 類型（分割任務）
            if shape_type != "polygon":
                continue

            if label not in class_mapping:
                print(f"[Notice] 跳過未定義的類別 '{label}' (位於 {json_path.name})")
                continue

            class_id = class_mapping[label]
            points = shape.get("points", [])

            if len(points) < 3:
                # 少於 3 個點無法構成多邊形
                continue

            # 歸一化座標 (x / img_w, y / img_h)
            normalized_points = []
            for x, y in points:
                norm_x = max(0.0, min(1.0, x / img_w))
                norm_y = max(0.0, min(1.0, y / img_h))
                normalized_points.extend([f"{norm_x:.6f}", f"{norm_y:.6f}"])

            line = f"{class_id} " + " ".join(normalized_points)
            yolo_lines.append(line)

        # 寫入對應的 txt 檔案
        txt_filename = f"{json_path.stem}.txt"
        txt_path = output_dir / txt_filename

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

        converted_count += 1

    print(f"轉換完成！共處理 {converted_count}/{len(json_files)} 個 JSON 檔案。")


# ==================== 使用範例 ====================
if __name__ == "__main__":
    # 1. 定義你的類別名稱對應 (Class Name -> Class ID)
    # 請根據你自己的 Labelme 標籤名稱進行調整
    CLASS_MAPPING = {
        "mf": 0,
        "ian": 1,
    }

    # 2. 設定路徑
    JSON_FOLDER = "dental_project/20261008/test"    
    OUTPUT_FOLDER = "runs/gt/labels"  

    # 執行轉換
    convert_labelme_to_yolov8_seg(
        json_dir=JSON_FOLDER,
        output_dir=OUTPUT_FOLDER,
        class_mapping=CLASS_MAPPING
    )