import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sklearn.cluster import KMeans
from scipy.spatial import distance
import glob

# ==========================================
# Module 1: YOLO Segmentation & FDI Logic
# ==========================================

def filter_isolated_teeth(candidates, distance_threshold_factor=4.0):
    if len(candidates) < 3: return candidates
    coords = np.array([(t['cx'], t['cy']) for t in candidates])
    dist_matrix = distance.cdist(coords, coords, 'euclidean')
    np.fill_diagonal(dist_matrix, np.inf)
    min_dists = dist_matrix.min(axis=1)
    threshold = np.median(min_dists) * distance_threshold_factor
    return [candidates[i] for i, dist in enumerate(min_dists) if dist <= threshold]

def calculate_iou(t1, t2):
    # Each mask is non-zero only inside its polygon's bounding box, so the intersection
    # only needs the overlap of the two boxes; union = |A| + |B| - |A & B|.
    # Same result as comparing the full-size masks, without scanning the whole image.
    ax1, ay1, ax2, ay2 = t1['bbox']
    bx1, by1, bx2, by2 = t2['bbox']
    x1, y1, x2, y2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    intersection = 0
    if x1 < x2 and y1 < y2:
        intersection = np.count_nonzero(t1['mask'][y1:y2, x1:x2] & t2['mask'][y1:y2, x1:x2])
    union = t1['pixels'] + t2['pixels'] - intersection
    return 0.0 if union == 0 else intersection / union

def process_and_separate_jaws(img_array, model):
    H, W, _ = img_array.shape
    results = model(img_array, retina_masks=True, verbose=False)
    
    if not results or not results[0].masks: 
        return [], []

    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    masks_xy = results[0].masks.xy 

    candidates = []
    areas = []

    for i, poly in enumerate(masks_xy):
        if len(poly) == 0: continue
            
        pts = poly.astype(np.int32)
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        bx, by, bw, bh = cv2.boundingRect(pts)
        bbox = (max(bx, 0), max(by, 0), min(bx + bw, W), min(by + bh, H))

        x1, y1, x2, y2 = map(int, boxes[i])
        margin = 5
        is_touching_border = (x1 < margin or y1 < margin or x2 > W - margin or y2 > H - margin)

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        y_max = y2  
        area = cv2.contourArea(poly)
        
        if area < 10: continue

        candidates.append({
            "id": i, "mask": mask, "bbox": bbox,
            "pixels": np.count_nonzero(mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]),
            "cx": cx, "cy": cy, "area": area,
            "y_max": y_max, "is_touching_border": is_touching_border, "score": confs[i]
        })
        areas.append(area)

    if not candidates: return [], []

    median_area = np.median(areas)
    layer1_passed = sorted(
        [t for t in candidates if not t['is_touching_border'] and median_area * 0.10 <= t['area'] <= median_area * 5.0],
        key=lambda x: x['score'], reverse=True
    )

    iou_filtered = []
    for current_tooth in layer1_passed:
        if not any(calculate_iou(current_tooth, saved) > 0.45 for saved in iou_filtered):
            iou_filtered.append(current_tooth)

    valid_teeth = filter_isolated_teeth(iou_filtered, distance_threshold_factor=4.0)
    if len(valid_teeth) < 2: return [], []

    y_coords = np.array([t['cy'] for t in valid_teeth]).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(y_coords)

    upper_label, lower_label = (0, 1) if kmeans.cluster_centers_[0] < kmeans.cluster_centers_[1] else (1, 0)
    
    lower_jaw = sorted([valid_teeth[i] for i, l in enumerate(labels) if l == lower_label], key=lambda x: x['cx'])
    upper_jaw = sorted([valid_teeth[i] for i, l in enumerate(labels) if l == upper_label], key=lambda x: x['cx'])

    return upper_jaw, lower_jaw

def assign_fdi_indices(sorted_teeth):
    if not sorted_teeth: return []
    num_teeth = len(sorted_teeth)
    
    if num_teeth < 4:
        mid_x = (sorted_teeth[0]['cx'] + sorted_teeth[-1]['cx']) / 2.0
    else:
        start_idx, end_idx = int(num_teeth * 0.1), int(num_teeth * 0.9) - 3 
        if end_idx < start_idx: start_idx, end_idx = 0, num_teeth - 3
            
        # 2026-10-25: 只考慮兩側都不超過 8 顆的位置(原本會把中線找偏,編出 39、410 這種不存在的編號);
        # 範圍內都不合法時才放寬到全部位置,仍然沒有就維持原本的做法。只影響原本一定編錯的影像(fdi_v2_check.py)
        def best_window(indices, limit):
            best, best_area = None, float('inf')
            for i in indices:
                if limit and (i + 2 > 8 or num_teeth - i - 2 > 8):
                    continue
                area_sum = sum(sorted_teeth[j]['area'] for j in range(i, i + 4))
                if area_sum < best_area:
                    best, best_area = i, area_sum
            return best

        best_4_idx = best_window(range(start_idx, end_idx + 1), True)
        if best_4_idx is None:
            best_4_idx = best_window(range(0, num_teeth - 3), True)
        if best_4_idx is None:
            best_4_idx = best_window(range(start_idx, end_idx + 1), False)
        mid_x = (sorted_teeth[best_4_idx + 1]['cx'] + sorted_teeth[best_4_idx + 2]['cx']) / 2.0

    for tooth in sorted_teeth:
        tooth['quadrant'] = 4 if tooth['cx'] <= mid_x else 3

    left_side = sorted([t for t in sorted_teeth if t['cx'] <= mid_x], key=lambda x: x['cx'], reverse=True) 
    for i, t in enumerate(left_side): t['fdi_id'] = f"{t['quadrant']}{i + 1}"

    right_side = sorted([t for t in sorted_teeth if t['cx'] > mid_x], key=lambda x: x['cx']) 
    for i, t in enumerate(right_side): t['fdi_id'] = f"{t['quadrant']}{i + 1}"

    return sorted(sorted_teeth, key=lambda x: x['cx'])

# ==========================================
# Module 2: In-Memory ROI Calculation (SPLIT LEFT/RIGHT)
# ==========================================

def get_split_roi_bounds(lower_teeth):
    """Calculates independent cropping bounds for the Left and Right sides."""
    bounds = {}
    sides = {
        'Right': {'targets': ['44', '45', '46'], 'premolars': ['44', '45']},
        'Left':  {'targets': ['34', '35', '36'], 'premolars': ['34', '35']}
    }
    MARGIN_X = 150

    for side_name, ids in sides.items():
        side_teeth = [t for t in lower_teeth if t.get('fdi_id') in ids['targets']]
        side_premolars = [t for t in lower_teeth if t.get('fdi_id') in ids['premolars']]
        
        if not side_premolars or not side_teeth: continue
            
        y_upper = int(min(t['cy'] for t in side_premolars))
        y_max_root = int(max(t['y_max'] for t in side_premolars))
        
        # Calculate the average height of the premolars on this side
        avg_tooth_height = int(np.mean([(t['y_max'] - (t['cy'] - (t['y_max'] - t['cy']))) for t in side_premolars]))
        
        x_min = max(0, int(min(t['cx'] for t in side_teeth)) - MARGIN_X)
        x_max = int(max(t['cx'] for t in side_teeth)) + MARGIN_X
        
        bounds[side_name] = {
            'y_upper': y_upper,
            'y_max_root': y_max_root,
            'x_min': x_min,
            'x_max': x_max,
            'dynamic_fallback': int(avg_tooth_height * 1.2) # Set fallback to 1.2x the tooth height
        }
        
    return bounds

def find_mandibular_border(img_array, premolar_y_max, search_x_min, search_x_max, dynamic_fallback):
    """
    Finds the jawline using Adaptive Peak Suppression. If the jawline is missing,
    it falls back to a distance proportional to the patient's actual tooth size.
    """
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_array
    roi = gray[premolar_y_max:, search_x_min:search_x_max]
    
    # If the image ends right at the root tip, use the proportional fallback
    if roi.shape[0] < 50 or roi.shape[1] < 10: 
        return min(img_array.shape[0], premolar_y_max + dynamic_fallback) 
        
    blurred = cv2.GaussianBlur(roi, (51, 51), 0)
    sobel_y = np.abs(cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5))
    sobel_y[:15, :] = 0 
    
    profile = np.sum(sobel_y, axis=1)
    first_peak_idx = np.argmax(profile)
    artifact_zone_start = int(roi.shape[0] * 0.80)
    
    if first_peak_idx > artifact_zone_start:
        suppression_start = max(0, first_peak_idx - 40) 
        profile[suppression_start:] = 0
        final_peak_idx = np.argmax(profile)
    else:
        final_peak_idx = first_peak_idx
        
    # Check if the detected peak is just background noise
    if profile[final_peak_idx] < (search_x_max - search_x_min) * 50: 
         # Jawline is missing. Use the patient-specific tooth height multiplier.
         return min(img_array.shape[0], premolar_y_max + dynamic_fallback)
         
    return final_peak_idx + premolar_y_max

# ==========================================
# Module 3: Export Helpers (Optional Debug Data)
# ==========================================

def draw_fdi_overlay(img, lower_jaw):
    vis_img = img.copy()
    for t in lower_jaw:
        cv2.drawContours(vis_img, [cv2.findContours(t['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]], -1, (0, 220, 100), 2)
        cv2.circle(vis_img, (t['cx'], t['cy']), 4, (0, 0, 255), -1)
        cv2.putText(vis_img, t.get('fdi_id', ''), (t['cx'] - 12, t['cy'] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    return vis_img

def export_csv(lower_teeth, csv_path):
    df = pd.DataFrame([{
        "FDI_ID": t.get('fdi_id', ''), "Center_X": t['cx'], 
        "Center_Y": t['cy'], "Y_Max": t['y_max'], "Area_px": int(t['area'])
    } for t in lower_teeth])
    df.to_csv(csv_path, index=False, encoding='utf-8')

# ==========================================
# Master Pipeline Orchestrator
# ==========================================

if __name__ == "__main__":
    MODEL_PATH = 'dental_project/train_success_replica3/weights/best.pt'
    INPUT_DIR = 'dental_project/20260826__mf_851'
    
    OUTPUT_DEBUG_IMG = 'dental_project/20260826__mf_851/output/visual_masks'
    OUTPUT_CSV = 'dental_project/20260826__mf_851/output/tooth_coordinates'
    OUTPUT_FINAL_ROI = 'dental_project/20260826__mf_851/output/roi_cropped'
    
    for folder in [OUTPUT_DEBUG_IMG, OUTPUT_CSV, OUTPUT_FINAL_ROI]:
        os.makedirs(folder, exist_ok=True)

    print("Loading YOLOv8 model...")
    model = YOLO(MODEL_PATH)
    image_list = glob.glob(os.path.join(INPUT_DIR, "*.[jJ][pP][gG]")) + glob.glob(os.path.join(INPUT_DIR, "*.[pP][nN][gG]"))
    
    success = 0
    for img_path in image_list:
        base_name = os.path.basename(img_path)
        stem = os.path.splitext(base_name)[0]
        
        try:
            img = cv2.imread(img_path)
            if img is None: continue
            
            upper_teeth, lower_teeth = process_and_separate_jaws(img, model)
            if not lower_teeth: continue
                
            lower_teeth = assign_fdi_indices(lower_teeth)
            
            # Export Debugs
            export_csv(lower_teeth, os.path.join(OUTPUT_CSV, f"{stem}_coords_fdi.csv"))
            cv2.imwrite(os.path.join(OUTPUT_DEBUG_IMG, f"{stem}_fdi.jpg"), draw_fdi_overlay(img, lower_teeth))
            
            # ----------------------------------------------------
            # The Split Execution
            # ----------------------------------------------------
            side_bounds = get_split_roi_bounds(lower_teeth)
            
            for side, b in side_bounds.items():
                
                # FIXED: Added b['dynamic_fallback'] to the function call
                jawline_y = find_mandibular_border(
                    img, 
                    b['y_max_root'], 
                    b['x_min'], 
                    b['x_max'],
                    b['dynamic_fallback']
                )
                
                h, w = img.shape[:2]
                jawline_y = min(jawline_y, h)
                x_max = min(b['x_max'], w)
                
                # Slicing the tensor
                final_roi = img[b['y_upper']:jawline_y, b['x_min']:x_max]
                
                # Save as separate files
                out_name = f"{stem}_{side}_roi.jpg"
                cv2.imwrite(os.path.join(OUTPUT_FINAL_ROI, out_name), final_roi)
            
            success += 1
            print(f"[Success] Cropped Left/Right ROIs for {base_name}")
            
        except Exception as e:
            print(f"[Error] Failed on {base_name}: {e}")

    print(f"\nUnified Pipeline Complete: {success} patients processed.")