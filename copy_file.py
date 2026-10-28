import os
import shutil
import filecmp
from pathlib import Path

def copy_identical_files(dir_a: str, dir_b: str, output_dir: str):
    path_a = Path(dir_a)
    path_b = Path(dir_b)
    dest_path = Path(output_dir)
    
    if not path_a.exists() or not path_b.exists():
        print("錯誤：來源資料夾不存在！")
        return

    # 建立目標資料夾（若不存在）
    dest_path.mkdir(parents=True, exist_ok=True)
    
    matched_count = 0

    # 遍歷資料夾 A 中的所有檔案
    for root, _, files in os.walk(path_a):
        for file in files:
            file_a = Path(root) / file
            
            # 取得相對於資料夾 A 的相對路徑
            relative_path = file_a.relative_to(path_a)
            file_b = path_b / relative_path
            
            # 只複製兩個資料夾中具有相同相對路徑的影像檔案
            if file_b.is_file() and file_a.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                target_file = dest_path / relative_path
                
                # 確保目標子資料夾存在
                target_file.parent.mkdir(parents=True, exist_ok=True)
                
                # 複製檔案並保留原始中繼資料（修改時間等）
                shutil.copy2(file_a, target_file)
                print(f"複製同名檔案: {relative_path}")
                matched_count += 1

    print(f"\n複製完成！共複製 {matched_count} 個同名影像檔案至: {output_dir}")

# --- 設定路徑並執行 ---
if __name__ == "__main__":
    DIR_A = "dental_project/20260909/mf_297/images"      # 第一個資料夾路徑
    DIR_B = "dental_project/20260902/20260901_mf_341_original/images"      # 第二個資料夾路徑
    OUTPUT_DIR = "dental_project/20260909/mf_297_original/images"   # 存放相同檔案的第三個資料夾

    copy_identical_files(DIR_A, DIR_B, OUTPUT_DIR)