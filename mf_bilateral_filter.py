import os
from pathlib import Path
import cv2

def batch_bilateral_filter(
    input_folder: str,
    output_folder: str,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
    valid_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
) -> None:
    """
    批次對指定資料夾內的圖片套用雙邊濾波（Bilateral Filter）。

    :param input_folder: 輸入圖片資料夾路徑
    :param output_folder: 輸出處理後圖片的資料夾路徑
    :param d: 濾波直徑（各像素領域的直徑，建議設 5~9；設為 -1 則自動由 sigma_space 計算）
    :param sigma_color: 顏色空間的標準差（值越大，顏色差異較大的像素也會融合在一起）
    :param sigma_space: 坐標空間的標準差（值越大，較遠的像素也會相互影響）
    :param valid_extensions: 支援處理的圖片副檔名
    """
    in_path = Path(input_folder)
    out_path = Path(output_folder)

    if not in_path.exists() or not in_path.is_dir():
        raise ValueError(f"輸入資料夾不存在或非有效目錄: {input_folder}")

    # 確保輸出資料夾存在
    out_path.mkdir(parents=True, exist_ok=True)

    # 取得所有匹配副檔名的檔案
    image_files = [
        f for f in in_path.iterdir() 
        if f.is_file() and f.suffix.lower() in valid_extensions
    ]

    if not image_files:
        print(f"在 {input_folder} 中未找到支援的圖片檔案。")
        return

    print(f"找到 {len(image_files)} 張圖片，開始進行雙邊濾波處理...")

    success_count = 0
    for img_path in image_files:
        # 讀取圖片（含路徑相容性處理）
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"跳過無法讀取的檔案: {img_path.name}")
            continue

        # 套用雙邊濾波
        filtered_img = cv2.bilateralFilter(
            src=img,
            d=d,
            sigmaColor=sigma_color,
            sigmaSpace=sigma_space
        )

        # 輸出結果
        output_file_path = out_path / img_path.name
        cv2.imwrite(str(output_file_path), filtered_img)
        success_count += 1
        print(f"[完成] {img_path.name} -> {output_file_path.name}")

    print(f"\n處理完成！成功轉換: {success_count}/{len(image_files)} 張圖片。")


if __name__ == "__main__":
    # 使用範例
    INPUT_DIR = "dental_project/20260912/mf_559_original/images"
    OUTPUT_DIR = "dental_project/20260913/mf_559_bilateral_filter/images"

    batch_bilateral_filter(
        input_folder=INPUT_DIR,
        output_folder=OUTPUT_DIR,
        d=5,               # 領域直徑
        sigma_color=50.0,  # 顏色平滑度
        sigma_space=50.0   # 空間平滑度
    )