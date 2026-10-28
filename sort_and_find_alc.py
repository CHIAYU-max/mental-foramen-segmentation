import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
from scipy.spatial import distance
import os
from pathlib import Path

# ==========================================
# 第一部分：資料清理與排序 (Preprocessing & Sorting)
# ==========================================

def filter_isolated_teeth(candidates, distance_threshold_factor=3.0):
    """
    過濾孤立點：如果某個物件離最近的鄰居太遠，視為雜訊。
    """
    if len(candidates) < 3:
        return candidates

    coords = np.array([(t['cx'], t['cy']) for t in candidates])
    dist_matrix = distance.cdist(coords, coords, 'euclidean')
    np.fill_diagonal(dist_matrix, np.inf)

    min_dists = dist_matrix.min(axis=1)
    median_dist = np.median(min_dists)
    threshold = median_dist * distance_threshold_factor

    print("-" * 30)
    print(f"孤立點偵測: 平均鄰居距離={median_dist:.1f} pixel")
    print(f"孤立容忍門檻: {threshold:.1f} pixel (超過此距離視為雜訊)")
    print("-" * 30)

    filtered_candidates = []
    for i, dist in enumerate(min_dists):
        if dist <= threshold:
            filtered_candidates.append(candidates[i])
        else:
            print(f"-> [移除] ID:{candidates[i]['id']} (原因: 孤立雜訊, 距離鄰居 {dist:.1f} px)")

    return filtered_candidates


def process_and_separate_jaws_ultimate(image_path, model):
    img = cv2.imread(image_path)
    if img is None:
        print(f"錯誤：無法讀取圖片 {image_path}")
        return None, None, None

    H, W, _ = img.shape

    results = model(img, retina_masks=True, verbose=False)
    if not results[0].masks: return None, None, None

    raw_masks = results[0].masks.data.cpu().numpy()
    candidates = []
    areas = []

    # --- 步驟 1: 基礎收集 ---
    for i, mask_tensor in enumerate(raw_masks):
        mask = mask_tensor.astype(np.uint8)
        if mask.shape[:2] != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        # 邊界檢查
        bbox = cv2.boundingRect(mask)
        x, y, w, h = bbox
        margin = 5
        is_touching_border = (x < margin or y < margin or x + w > W - margin or y + h > H - margin)

        moments = cv2.moments(mask)
        area = moments["m00"]
        if area < 10: continue

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        candidates.append({
            "id": i, "mask": mask, "cx": cx, "cy": cy, "area": area,
            "is_touching_border": is_touching_border
        })
        areas.append(area)

    if not candidates: return img, [], []

    # --- 步驟 2: 面積 + 邊界過濾 ---
    median_area = np.median(areas)
    lower_area = median_area * 0.15
    upper_area = median_area * 4.0

    layer1_passed = []
    for t in candidates:
        if t['is_touching_border']:
            print(f"-> [移除] ID:{t['id']} (原因: 接觸邊界)")
            continue
        if lower_area <= t['area'] <= upper_area:
            layer1_passed.append(t)
        else:
            print(f"-> [移除] ID:{t['id']} (原因: 面積異常 {t['area']:.0f})")

    if not layer1_passed: return img, [], []

    # --- 步驟 3: 孤立點過濾 ---
    valid_teeth = filter_isolated_teeth(layer1_passed, distance_threshold_factor=4.0)

    if len(valid_teeth) < 2: return img, [], []

    # --- 步驟 4: 分排與排序 ---
    y_coords = np.array([t['cy'] for t in valid_teeth]).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(y_coords)

    centers = kmeans.cluster_centers_
    if centers[0] < centers[1]:
        upper_label, lower_label = 0, 1
    else:
        upper_label, lower_label = 1, 0

    # 依照 X 座標 (由左至右) 排序
    upper_jaw = sorted([valid_teeth[i] for i, l in enumerate(labels) if l == upper_label], key=lambda x: x['cx'])
    lower_jaw = sorted([valid_teeth[i] for i, l in enumerate(labels) if l == lower_label], key=lambda x: x['cx'])

    # ==========================================
    # --- [新增] 重新賦予由左至右的連續 ID ---
    # ==========================================
    for i, tooth in enumerate(upper_jaw):
        tooth['id'] = f"U{i + 1}"  # 上排牙齒重編為 U1, U2, U3...

    for i, tooth in enumerate(lower_jaw):
        tooth['id'] = f"L{i + 1}"  # 下排牙齒重編為 L1, L2, L3...

    return img, upper_jaw, lower_jaw


# ==========================================
# 第二部分：幾何分析與連線 (Geometric Analysis) - RANSAC 頂點對齊版
# ==========================================
import numpy as np
import cv2
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def apply_clahe(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced


def get_closest_contour_point(mask, target_point):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours: return None
    largest_contour = max(contours, key=cv2.contourArea)
    pts = largest_contour.squeeze()
    if pts.ndim == 1: pts = pts.reshape(-1, 2)
    dists = np.sum((pts - target_point) ** 2, axis=1)
    min_idx = np.argmin(dists)
    return tuple(pts[min_idx])


def scan_vertical_line_for_edge(img_roi, start_offset=0):
    grad_y = cv2.Sobel(img_roi, cv2.CV_64F, 0, 1, ksize=3)
    grad_y[grad_y < 0] = 0

    ignore_top = 10 + start_offset
    if len(grad_y) > ignore_top:
        grad_y[:ignore_top] = 0
    else:
        return 0

    center_y = len(grad_y) // 2
    sigma = len(grad_y) * 0.4
    gaussian_weight = np.exp(-((np.arange(len(grad_y)) - center_y) ** 2) / (2 * sigma ** 2))

    weighted_grad = grad_y.flatten() * gaussian_weight
    best_y = np.argmax(weighted_grad)

    if weighted_grad[best_y] < 5: return None
    return best_y


def find_bone_height_voting_raw(enhanced_img, tooth_left, tooth_right):
    H, W = enhanced_img.shape
    ys_l, xs_l = np.where(tooth_left['mask'] > 0)
    ys_r, xs_r = np.where(tooth_right['mask'] > 0)
    if len(xs_l) == 0 or len(xs_r) == 0: return None

    right_edge_l = np.max(xs_l)
    left_edge_r = np.min(xs_r)
    overlap_width = right_edge_l - left_edge_r
    is_overlapping = overlap_width > 0

    intersection = cv2.bitwise_and(tooth_left['mask'], tooth_right['mask'])
    overlap_count = cv2.countNonZero(intersection)

    # 取得左右兩顆牙齒的中心點 Y 座標平均值
    avg_cy = int((tooth_left['cy'] + tooth_right['cy']) / 2)
    
    # 🌟 修正：鎖定腰部範圍！
    # 只在中心點往上 40 pixel 到往下 60 pixel 的「極窄區域」內搜尋
    # 這樣它既看不到牙冠頂部，也看不到牙根底部
    search_top = max(0, avg_cy - 40)
    search_bottom = min(H, avg_cy + 60)
    offset_overlap = 0

    if overlap_count > 0:
        ys_int, _ = np.where(intersection > 0)
        if len(ys_int) > 0:
            search_top = max(0, np.min(ys_int) - 30) 
            search_bottom = min(H, np.min(ys_int) + 50)
            offset_overlap = 5

    search_bottom = min(H, search_top + 250)
    search_top = min(search_top, H - 50)

    detected_ys = []

    if not is_overlapping:
        gap_start, gap_end = right_edge_l, left_edge_r
        scan_xs = np.linspace(gap_start, gap_end, num=7, dtype=int)
        scan_xs = scan_xs[1:-1]
        for scan_x in scan_xs:
            scan_x = max(0, min(W - 1, scan_x))
            roi = enhanced_img[search_top:search_bottom, max(0, scan_x - 1):min(W, scan_x + 2)]
            roi = np.mean(roi, axis=1)
            res = scan_vertical_line_for_edge(roi, 0)
            if res is not None: detected_ys.append(search_top + res)
    else:
        mid_x = int((right_edge_l + left_edge_r) / 2)
        mid_x = max(0, min(W - 1, mid_x))
        roi = enhanced_img[search_top:search_bottom, max(0, mid_x - 2):min(W, mid_x + 3)]
        roi = np.mean(roi, axis=1)
        res = scan_vertical_line_for_edge(roi, offset_overlap)
        if res is not None: detected_ys.append(search_top + res)

    if not detected_ys: return None
    return int(np.median(detected_ys))


from scipy.optimize import least_squares
import numpy as np

def apply_ransac_and_shift_up(points_data, img_shape, degree=2):
    if len(points_data) < 3: 
        return points_data, None

    X = np.array([p[1] for p in points_data])
    Y = np.array([p[2] for p in points_data])

    def residuals(params, x, y):
        a, b, c = params
        return (a * x**2 + b * x + c) - y

    max_a = -5e-6 
    
    # 【修正 1】: 確保初始猜測值 p0[0] 符合 bounds，否則會當機！
    p0 = [max_a, 0.0, np.median(Y)]
    bounds = ([-np.inf, -np.inf, -np.inf], [max_a, np.inf, np.inf])

    try:
        res = least_squares(
            residuals, 
            p0, 
            args=(X, Y), 
            bounds=bounds, 
            loss='soft_l1', 
            f_scale=15.0
        )
        
        a, b, c = res.x

        Y_pred_all = a * X**2 + b * X + c
        diffs = Y - Y_pred_all
        
        # 【修正 2】: 調整 Shift Up 的力道
        # 這裡設定 percentile(diffs, 5)。
        # 後面的 - 15 是一個「手動提昇」的參數。如果橘線出來後，
        # 你覺得還是離牙冠太遠（太低），就把 -15 改成 -30 或 -40，強制它往上貼！
        shift_up_amount = np.percentile(diffs, 5)

        smoothed_data = []
        X_range = np.arange(0, img_shape[1])
        Y_curve_base = a * X_range**2 + b * X_range + c
        Y_curve_shifted = Y_curve_base + shift_up_amount
        curve_coords = np.column_stack((X_range, Y_curve_shifted))

        for i, (idx, mx, my) in enumerate(points_data):
            pred_y = a * mx**2 + b * mx + c
            final_y = int(pred_y + shift_up_amount)
            final_y = max(0, min(img_shape[0] - 1, final_y))
            smoothed_data.append((idx, mx, final_y))

        return smoothed_data, curve_coords

    except Exception as e:
        print(f"曲線擬合失敗: {e}")
        return points_data, None


def draw_alveolar_crest(img, lower_jaw):
    vis_img = img.copy()
    enhanced_img = apply_clahe(img)
    H, W = img.shape[:2]

    raw_points = []
    # 收集原始骨頭高度點
    for i in range(len(lower_jaw) - 1):
        left_tooth = lower_jaw[i]
        right_tooth = lower_jaw[i + 1]
        mid_x = int((left_tooth['cx'] + right_tooth['cx']) / 2)

        # 使用投票法找高度
        raw_y = find_bone_height_voting_raw(enhanced_img, left_tooth, right_tooth)

        if raw_y is not None:
            raw_points.append((i, mid_x, raw_y))

    # 使用 RANSAC + Shift Up 進行平滑與修正
    smooth_points, curve_coords = apply_ransac_and_shift_up(raw_points, img.shape, degree=2)

    # 畫橘色參考曲線 (RANSAC 擬合結果)
    if curve_coords is not None:
        pts_curve = curve_coords.astype(np.int32)
        cv2.polylines(vis_img, [pts_curve], False, (0, 100, 255), 2)

    # 建立平滑後的高度對照表
    smooth_map = {item[0]: item[2] for item in smooth_points}

    # 用來儲存座標的列表 [index, start_point, end_point]
    line_records = []

    for i in range(len(lower_jaw) - 1):
        if i in smooth_map:
            bone_y = smooth_map[i]
            left_tooth = lower_jaw[i]
            right_tooth = lower_jaw[i + 1]
            mid_x = int((left_tooth['cx'] + right_tooth['cx']) / 2)

            # 根據平滑後的高度，找牙齒輪廓上最近的點
            target_pt = np.array([mid_x, bone_y])
            pt1 = get_closest_contour_point(left_tooth['mask'], target_pt)  # 左邊牙齒的連線點
            pt2 = get_closest_contour_point(right_tooth['mask'], target_pt)  # 右邊牙齒的連線點

            if pt1 is not None and pt2 is not None:
                # 畫圖 (綠色線)
                cv2.circle(vis_img, pt1, 5, (255, 0, 0), -1)
                cv2.circle(vis_img, pt2, 5, (255, 0, 0), -1)
                cv2.line(vis_img, pt1, pt2, (0, 255, 0), 2)

                # --- [新增] 儲存座標數據 ---
                # 記錄格式: (間隙編號, 左點座標, 右點座標)
                line_records.append({
                    "gap_index": i,
                    "left_tooth_id": left_tooth['id'],
                    "right_tooth_id": right_tooth['id'],
                    "p1": pt1,  # (x, y)
                    "p2": pt2  # (x, y)
                })

    return vis_img, line_records, curve_coords


def save_lines_to_txt(records, filename):
    """
    將線段座標寫入文字檔
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(
            f"{'Gap_Index':<10} {'Left_Tooth_ID':<15} {'Right_Tooth_ID':<15} {'Start_Point(x,y)':<20} {'End_Point(x,y)':<20}\n")
        f.write("=" * 85 + "\n")

        for rec in records:
            p1_str = f"{rec['p1']}"
            p2_str = f"{rec['p2']}"
            f.write(
                f"{rec['gap_index']:<10} {rec['left_tooth_id']:<15} {rec['right_tooth_id']:<15} {p1_str:<20} {p2_str:<20}\n")

    print(f"座標已儲存至: {filename}")

import os
import numpy as np

def export_yolo_pose_labels(teeth, curve_coords, img_shape, output_txt_path, class_id=0):
    """
    將牙齒遮罩與計算出的齒槽骨脊(ALC)曲線，轉換為 YOLO-Pose 訓練格式
    每個標註包含: BBox (中心點, 寬, 高) + 2 個 Keypoints (左側 ALC 點, 右側 ALC 點)
    """
    H, W = img_shape[:2]
    
    # 將曲線座標建立成字典，方便用 X 快速查 Y 高度
    if curve_coords is not None:
        curve_dict = {int(pt[0]): pt[1] for pt in curve_coords}
    else:
        curve_dict = {}

    lines = []
    for tooth in teeth:
        mask = tooth['mask']
        ys, xs = np.where(mask > 0)
        if len(xs) == 0: continue

        # ==========================================
        # 1. 萃取 Bounding Box 並正規化
        # ==========================================
        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)
        
        w_box = x_max - x_min
        h_box = y_max - y_min
        x_c = x_min + w_box / 2.0
        y_c = y_min + h_box / 2.0

        # 正規化至 0.0 ~ 1.0
        nx_c, ny_c = x_c / W, y_c / H
        nw, nh = w_box / W, h_box / H

        # ==========================================
        # 2. 尋找這顆牙齒的左、右兩個 ALC 關鍵點
        # ==========================================
        # 左側關鍵點：拿牙齒最左邊的 X 去查曲線的 Y，並貼合到牙齒輪廓上
        left_y = curve_dict.get(x_min, tooth['cy']) 
        pt_left = get_closest_contour_point(mask, np.array([x_min, left_y]))
        
        # 右側關鍵點：拿牙齒最右邊的 X 去查曲線的 Y，並貼合到牙齒輪廓上
        right_y = curve_dict.get(x_max, tooth['cy'])
        pt_right = get_closest_contour_point(mask, np.array([x_max, right_y]))

        # ==========================================
        # 3. 關鍵點正規化與可見度標記
        # ==========================================
        # v=2 表示點位可見且標記
        if pt_left is not None:
            nkp1_x, nkp1_y, nkp1_v = pt_left[0] / W, pt_left[1] / H, 2
        else:
            nkp1_x, nkp1_y, nkp1_v = 0.0, 0.0, 0

        if pt_right is not None:
            nkp2_x, nkp2_y, nkp2_v = pt_right[0] / W, pt_right[1] / H, 2
        else:
            nkp2_x, nkp2_y, nkp2_v = 0.0, 0.0, 0

        # ==========================================
        # 4. 組合成 YOLO-Pose 的文字行
        # ==========================================
        line = f"{class_id} {nx_c:.6f} {ny_c:.6f} {nw:.6f} {nh:.6f} {nkp1_x:.6f} {nkp1_y:.6f} {nkp1_v} {nkp2_x:.6f} {nkp2_y:.6f} {nkp2_v}"
        lines.append(line)

    # 寫入文字檔
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
# ==========================================
# 主程式
# ==========================================
# ... 前面所有的 import 和 def 保持不變 ...

# ==========================================
# 主程式：批次處理資料夾
# ==========================================
# ==========================================
# 主程式：批次處理資料夾並輸出標準 YOLO 格式
# ==========================================
import os
import shutil # 用於複製原圖

# ==========================================
# 主程式：批次處理資料夾並輸出標準 YOLO 格式與視覺化檢查圖
# ==========================================
if __name__ == "__main__":
    # --- 設定輸入路徑 ---
    MODEL_PATH = 'dental_project/train_success_replica3/weights/best.pt'
    IMAGE_FOLDER = 'datasets/20251217_alc_66/test/images'
    
    # --- 設定輸出路徑 ---
    OUTPUT_DIR = 'runs/test/alc_lines'
    
    # 🌟 新增：定義三個子資料夾路徑
    #IMAGES_ORIGINAL_DIR = os.path.join(OUTPUT_DIR, 'images_original')
    IMAGES_ANNOTATED_DIR = os.path.join(OUTPUT_DIR, 'images_annotated')
    LABELS_OUT_DIR = os.path.join(OUTPUT_DIR, 'labels')
    COORDS_OUT_DIR = os.path.join(OUTPUT_DIR, 'coords_txt')

    # 🌟 建立資料夾 (會自動建立父目錄)
    #os.makedirs(IMAGES_ORIGINAL_DIR, exist_ok=True)
    os.makedirs(IMAGES_ANNOTATED_DIR, exist_ok=True)
    os.makedirs(LABELS_OUT_DIR, exist_ok=True)
    os.makedirs(COORDS_OUT_DIR, exist_ok=True)

    # 1. 預先載入模型
    if not os.path.exists(MODEL_PATH):
        print(f"錯誤：找不到模型檔案 {MODEL_PATH}")
    else:
        model = YOLO(MODEL_PATH)
        print("模型載入成功，開始處理圖片...")

        # 2. 遍歷資料夾
        image_extensions = ('.jpg', '.png', '.jpeg', '.JPG', '.PNG')
        all_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(image_extensions)]
        
        if not all_files:
            print(f"警告：在 {IMAGE_FOLDER} 中找不到任何圖片。")

        for img_file in all_files:
            img_path = os.path.join(IMAGE_FOLDER, img_file)
            file_stem = os.path.splitext(img_file)[0]  # 取得不含副檔名的主檔名
            
            print(f"\n[處理中] {img_file}...")

            try:
                # 執行推理與分群
                img, upper_teeth, lower_teeth = process_and_separate_jaws_ultimate(img_path, model)
                
                if img is not None and lower_teeth and len(lower_teeth) >= 2:
                    # 執行幾何分析與 ALC 曲線計算
                    final_result, line_data, curve_coords = draw_alveolar_crest(img, lower_teeth)

                    # ==========================================
                    # 🌟 分流存檔邏輯 (分為三類)
                    # ==========================================
                    
                    # 1. 將「原始圖片」存入 images_original (直接複製檔案最快且保真)
                   # orig_img_output_path = os.path.join(IMAGES_ORIGINAL_DIR, f"{file_stem}.jpg")
                    #shutil.copy2(img_path, orig_img_output_path)

                    # 2. 將「畫有線條的結果圖」存入 images_annotated (檢查用，可以加後綴避免混淆)
                    annotated_img_output_path = os.path.join(IMAGES_ANNOTATED_DIR, f"{file_stem}_annotated.jpg")
                    cv2.imwrite(annotated_img_output_path, final_result)

                    # 3. 將「YOLO 標註檔」存入 labels 子資料夾
                    yolo_txt_path = os.path.join(LABELS_OUT_DIR, f"{file_stem}.txt")
                    
                    export_yolo_pose_labels(
                        teeth=lower_teeth, 
                        curve_coords=curve_coords, 
                        img_shape=img.shape, 
                        output_txt_path=yolo_txt_path,
                        class_id=0
                    )
                    coords_txt_path = os.path.join(COORDS_OUT_DIR, f"{file_stem}.txt")
                    save_lines_to_txt(line_data, coords_txt_path)

                    print(f"成功：已輸出 原圖、標註圖、TXT 至對應資料夾")
                else:
                    print(f"跳過：{img_file} 未偵測到足夠下排牙齒。")
                    
            except Exception as e:
                print(f"處理檔案 {img_file} 時發生錯誤: {e}")
        
    print("\n所有任務已完成。")