import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sklearn.cluster import KMeans
from scipy.spatial import distance

# ==========================================
# 核心演算法：過濾與分排
# ==========================================

def filter_isolated_teeth(candidates, distance_threshold_factor=4.0):
    if len(candidates) < 3:
        return candidates

    coords = np.array([(t['cx'], t['cy']) for t in candidates])
    dist_matrix = distance.cdist(coords, coords, 'euclidean')
    np.fill_diagonal(dist_matrix, np.inf)

    min_dists = dist_matrix.min(axis=1)
    median_dist = np.median(min_dists)
    threshold = median_dist * distance_threshold_factor

    filtered_candidates = []
    for i, dist in enumerate(min_dists):
        if dist <= threshold:
            filtered_candidates.append(candidates[i])
    return filtered_candidates

def calculate_iou(t1, t2):
    # 遮罩只在自己的外框內有值，交集只需比對兩個外框重疊的區域，聯集 = |A| + |B| - |A ∩ B|；
    # 結果與整張圖逐像素比對相同，但不必每次掃過整張全景片
    ax1, ay1, ax2, ay2 = t1['bbox']
    bx1, by1, bx2, by2 = t2['bbox']
    x1, y1, x2, y2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    intersection = 0
    if x1 < x2 and y1 < y2:
        intersection = np.count_nonzero(t1['mask'][y1:y2, x1:x2] & t2['mask'][y1:y2, x1:x2])
    union = t1['pixels'] + t2['pixels'] - intersection
    if union == 0:
        return 0.0
    return intersection / union

def process_and_separate_jaws(image_path, model):
    img = cv2.imread(image_path)
    if img is None:
        print(f"錯誤：無法讀取圖片 {image_path}")
        return None, None, None

    H, W, _ = img.shape
    results = model(img, retina_masks=True, verbose=False)
    
    if not results or not results[0].masks: 
        return img, [], []

    raw_masks = results[0].masks.data.cpu().numpy()
    candidates = []
    areas = []

    for i, mask_tensor in enumerate(raw_masks):
        mask = mask_tensor.astype(np.uint8)
        if mask.shape[:2] != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        bbox = cv2.boundingRect(mask)
        x, y, w, h = bbox
        margin = 5
        is_touching_border = (x < margin or y < margin or x + w > W - margin or y + h > H - margin)

        moments = cv2.moments(mask)
        area = moments["m00"]
        if area < 10: 
            continue

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        
        # 提取 y_max (Mask 的最底端)，用於後續尋找下顎邊界
        y_max = y + h

        candidates.append({
            "id": i, "mask": mask, "cx": cx, "cy": cy, "area": area,
            "y_max": y_max,
            "bbox": (x, y, x + w, y + h),
            "pixels": np.count_nonzero(mask[y:y + h, x:x + w]),
            "is_touching_border": is_touching_border,
            "score": results[0].boxes.conf[i].item() if results[0].boxes is not None else 1.0
        })
        areas.append(area)

    if not candidates: 
        return img, [], []

    median_area = np.median(areas)
    lower_area = median_area * 0.10
    upper_area = median_area * 5.0

    layer1_passed = []
    for t in candidates:
        if t['is_touching_border']:
            continue
        if lower_area <= t['area'] <= upper_area:
            layer1_passed.append(t)

    if not layer1_passed: 
        return img, [], []

    layer1_passed = sorted(layer1_passed, key=lambda x: x['score'], reverse=True)
    iou_filtered = []
    IOU_THRESHOLD = 0.45 

    for current_tooth in layer1_passed:
        keep = True
        for saved_tooth in iou_filtered:
            iou = calculate_iou(current_tooth, saved_tooth)
            if iou > IOU_THRESHOLD:
                keep = False
                break
        if keep:
            iou_filtered.append(current_tooth)

    if not iou_filtered:
        return img, [], []

    valid_teeth = filter_isolated_teeth(iou_filtered, distance_threshold_factor=4.0)
    if len(valid_teeth) < 2: 
        return img, [], []

    y_coords = np.array([t['cy'] for t in valid_teeth]).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(y_coords)

    centers = kmeans.cluster_centers_
    if centers[0] < centers[1]:
        upper_label, lower_label = 0, 1
    else:
        upper_label, lower_label = 1, 0

    lower_jaw = sorted([valid_teeth[i] for i, l in enumerate(labels) if l == lower_label], key=lambda x: x['cx'])
    upper_jaw = sorted([valid_teeth[i] for i, l in enumerate(labels) if l == upper_label], key=lambda x: x['cx'])

    # 重新給予連續的下排原始 ID (L1, L2...)
    for i, tooth in enumerate(lower_jaw):
        tooth['id'] = f"L{i + 1}"

    return img, upper_jaw, lower_jaw


# ==========================================
# FDI 編碼邏輯
# ==========================================

def assign_fdi_indices(sorted_teeth, jaw_type='lower', is_xray=True):
    """
    結合「群組面積」與「位置」策略：
    先鎖定連續 4 顆面積總和最小的牙齒 (即四顆門牙)，再從這四顆正中間切出生物學中線。
    """
    if not sorted_teeth:
        return []
        
    num_teeth = len(sorted_teeth)
    
    if num_teeth < 4:
        min_x = sorted_teeth[0]['cx']
        max_x = sorted_teeth[-1]['cx']
        mid_x = (min_x + max_x) / 2.0
    else:
        start_idx = int(num_teeth * 0.1)
        end_idx = int(num_teeth * 0.9) - 3 
        
        if end_idx < start_idx: 
            start_idx = 0
            end_idx = num_teeth - 3
            
        min_4_area = float('inf')
        best_4_idx = 0
        
        for i in range(start_idx, end_idx + 1):
            area_sum = sum(sorted_teeth[j]['area'] for j in range(i, i + 4))
            
            if area_sum < min_4_area:
                min_4_area = area_sum
                best_4_idx = i

        mid_left_tooth = sorted_teeth[best_4_idx + 1]
        mid_right_tooth = sorted_teeth[best_4_idx + 2]
        
        mid_x = (mid_left_tooth['cx'] + mid_right_tooth['cx']) / 2.0

    for tooth in sorted_teeth:
        if jaw_type == 'lower':
            if is_xray:
                tooth['quadrant'] = 4 if tooth['cx'] <= mid_x else 3
            else:
                tooth['quadrant'] = 3 if tooth['cx'] <= mid_x else 4
        else: # upper
            if is_xray:
                tooth['quadrant'] = 1 if tooth['cx'] <= mid_x else 2
            else:
                tooth['quadrant'] = 2 if tooth['cx'] <= mid_x else 1

    left_side_teeth = [t for t in sorted_teeth if t['cx'] <= mid_x]
    left_side_teeth = sorted(left_side_teeth, key=lambda x: x['cx'], reverse=True) 
    
    for i, tooth in enumerate(left_side_teeth):
        tooth['fdi_id'] = f"{tooth['quadrant']}{i + 1}"

    right_side_teeth = [t for t in sorted_teeth if t['cx'] > mid_x]
    right_side_teeth = sorted(right_side_teeth, key=lambda x: x['cx']) 
    
    for i, tooth in enumerate(right_side_teeth):
        tooth['fdi_id'] = f"{tooth['quadrant']}{i + 1}"

    return sorted(sorted_teeth, key=lambda x: x['cx'])


# ==========================================
# 視覺化與資料輸出功能 (支援雙模式)
# ==========================================

def draw_lower_teeth_only(img, lower_jaw, use_fdi=False):
    """
    use_fdi=True 顯示 FDI 編碼，use_fdi=False 顯示 L1, L2...
    """
    vis_img = img.copy()
    
    for tooth in lower_jaw:
        mask = tooth['mask']
        cx, cy = tooth['cx'], tooth['cy']
        
        tooth_id = tooth.get('fdi_id', '') if use_fdi else tooth.get('id', '')
        color = (0, 220, 100) 
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_img, contours, -1, color, 2)
        
        cv2.circle(vis_img, (cx, cy), 4, (0, 0, 255), -1)
        
        cv2.putText(vis_img, str(tooth_id), (cx - 12, cy - 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        
    return vis_img


def export_tooth_coordinates(lower_teeth, output_txt_path, output_csv_path, use_fdi=False):
    """
    根據 use_fdi 決定導出的 ID 標籤與內容，現已加入 Y_Max
    """
    records = []
    id_header = "FDI_ID" if use_fdi else "Tooth_ID"
    
    for tooth in lower_teeth:
        tooth_val = tooth.get('fdi_id', '') if use_fdi else tooth.get('id', '')
        records.append({
            id_header: tooth_val,
            "Center_X": tooth['cx'],
            "Center_Y": tooth['cy'],
            "Y_Max": tooth.get('y_max', 0),
            "Area_px": int(tooth['area'])
        })
        
    df = pd.DataFrame(records)
    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(f"{id_header:<12} {'Center_X':<12} {'Center_Y':<12} {'Y_Max':<12}\n")
        f.write("=" * 55 + "\n")
        for rec in records:
            f.write(f"{rec[id_header]:<12} {rec['Center_X']:<12} {rec['Center_Y']:<12} {rec['Y_Max']:<12}\n")


# ==========================================
# 主程式：資料夾批次處理
# ==========================================

if __name__ == "__main__":
    MODEL_PATH = 'dental_project/train_success_replica3/weights/best.pt'
    IMAGE_FOLDER = 'dental_project/20260826__mf_851'
    OUTPUT_DIR = 'dental_project/20260826__mf_851/output'
    
    IMAGES_OUT_DIR = os.path.join(OUTPUT_DIR, 'visual_masks')
    COORDS_OUT_DIR = os.path.join(OUTPUT_DIR, 'tooth_coordinates')
    
    os.makedirs(IMAGES_OUT_DIR, exist_ok=True)
    os.makedirs(COORDS_OUT_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print(f"錯誤：找不到模型檔案 {MODEL_PATH}")
    else:
        model = YOLO(MODEL_PATH)
        print("YOLO 模型載入成功，開始提取下排牙齒並進行雙模式輸出...")

        image_extensions = ('.jpg', '.png', '.jpeg', '.JPG', '.PNG')
        all_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(image_extensions)]
        
        for img_file in all_files:
            img_path = os.path.join(IMAGE_FOLDER, img_file)
            file_stem = os.path.splitext(img_file)[0]
            
            try:
                img, upper_teeth, lower_teeth = process_and_separate_jaws(img_path, model)
                
                if img is not None and lower_teeth:
                    lower_teeth = assign_fdi_indices(lower_teeth, jaw_type='lower', is_xray=True)
                    
                    # ---------------------------
                    # 1. 輸出原始排序 (L1, L2...)
                    # ---------------------------
                    vis_orig = draw_lower_teeth_only(img, lower_teeth, use_fdi=False)
                    cv2.imwrite(os.path.join(IMAGES_OUT_DIR, f"{file_stem}_lower_orig.jpg"), vis_orig)
                    
                    export_tooth_coordinates(
                        lower_teeth, 
                        os.path.join(COORDS_OUT_DIR, f"{file_stem}_coords_orig.txt"), 
                        os.path.join(COORDS_OUT_DIR, f"{file_stem}_coords_orig.csv"), 
                        use_fdi=False
                    )

                    # ---------------------------
                    # 2. 輸出 FDI 排序 (31, 41...)
                    # ---------------------------
                    vis_fdi = draw_lower_teeth_only(img, lower_teeth, use_fdi=True)
                    cv2.imwrite(os.path.join(IMAGES_OUT_DIR, f"{file_stem}_lower_fdi.jpg"), vis_fdi)
                    
                    export_tooth_coordinates(
                        lower_teeth, 
                        os.path.join(COORDS_OUT_DIR, f"{file_stem}_coords_fdi.txt"), 
                        os.path.join(COORDS_OUT_DIR, f"{file_stem}_coords_fdi.csv"), 
                        use_fdi=True
                    )
                    
                    print(f"[成功] {img_file} -> 已導出原始與 FDI 雙版本資料。")
                else:
                    print(f"[跳過] {img_file} -> 未能識別出有效的下排牙齒。")
                    
            except Exception as e:
                print(f"[錯誤] 處理檔案 {img_file} 時發生異常: {e}")
        
    print("\n雙模式座標導出已完成。")