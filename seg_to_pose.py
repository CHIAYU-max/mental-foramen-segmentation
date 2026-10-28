import os
import cv2
import numpy as np
from pathlib import Path

# 請在這裡修改輸入與輸出資料夾
INPUT_DIR = "dental_project/20260903/20260903_mf_341_unsharp_mask_brightness_0.2/labels"
OUTPUT_DIR = "dental_project/20260905/20260905_mf_341_unsharp_mask_brightness_0.2/labels"
IMG_W = 1280
IMG_H = 1280
NUM_KPTS = 6


def polygon_to_keypoints(pts_norm, img_w=1280, img_h=1280, num_kpts=6):
    """
    從長條 Polygon 提取有序的中心線關鍵點 (依 X 軸由後方延伸至頦孔)
    """
    pts = np.array(pts_norm).reshape(-1, 2)
    pts[:, 0] *= img_w
    pts[:, 1] *= img_h
    pts = pts.astype(np.int32)
    
    # 1. 建立二值 Mask
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    
    # 2. 找出神經管沿 X 方向的涵蓋範圍
    x_coords = np.where(mask.any(axis=0))[0]
    if len(x_coords) < num_kpts:
        return None
    
    # 3. 沿主方向等距取 N 個截面的幾何中心點
    sampled_x = np.linspace(x_coords.min(), x_coords.max(), num_kpts)
    keypoints = []
    
    for x in sampled_x:
        x_col = int(x)
        y_indices = np.where(mask[:, x_col] > 0)[0]
        if len(y_indices) > 0:
            y_center = np.mean(y_indices)
        else:
            # 若該欄剛好有斷點，取前後均值
            y_center = np.mean(np.where(mask > 0)[0])
        
        # 正規化座標，visibility 設為 2 (visible and labeled)
        keypoints.extend([x / img_w, y_center / img_h, 2.0])
        
    return keypoints

def convert_seg_to_pose_label(seg_txt_path, out_pose_txt_path, img_w=1280, img_h=1280, num_kpts=6):
    lines_out = []
    with open(seg_txt_path, "r") as f:
        for line in f.readlines():
            parts = list(map(float, line.strip().split()))
            cls_id = int(parts[0])
            poly_coords = parts[1:]
            
            # 取得 Bounding Box
            pts = np.array(poly_coords).reshape(-1, 2)
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
            w, h = (x_max - x_min), (y_max - y_min)
            
            if cls_id != 1:
                continue

            kpts = polygon_to_keypoints(poly_coords, img_w, img_h, num_kpts)
            if kpts is not None:
                # YOLO-Pose 格式: <class> <cx> <cy> <w> <h> <kpt1_x> <kpt1_y> <kpt1_v> ...
                kpt_str = " ".join([f"{v:.6f}" for v in kpts])
                lines_out.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {kpt_str}\n")
                
    with open(out_pose_txt_path, "w") as f:
        f.writelines(lines_out)


def convert_folder(input_dir, output_dir, img_w=1280, img_h=1280, num_kpts=6):
    """將輸入資料夾中的所有 YOLO segmentation labels 轉成 pose labels。"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.is_dir():
        raise NotADirectoryError(f"輸入資料夾不存在：{input_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    txt_files = sorted(input_path.glob("*.txt"))

    for seg_txt_path in txt_files:
        out_pose_txt_path = output_path / seg_txt_path.name
        convert_seg_to_pose_label(
            seg_txt_path,
            out_pose_txt_path,
            img_w=img_w,
            img_h=img_h,
            num_kpts=num_kpts,
        )

    print(f"已轉換 {len(txt_files)} 個標籤檔")


def main():
    convert_folder(
        INPUT_DIR,
        OUTPUT_DIR,
        img_w=IMG_W,
        img_h=IMG_H,
        num_kpts=NUM_KPTS,
    )


if __name__ == "__main__":
    main()