import os
import shutil
import random
from pathlib import Path

# ==================== 設定路徑與參數 ====================
SOURCE_DIR = Path("dental_project/20260913/mf_397_bf_gamma_clahe_um")  # 原始資料夾
TARGET_DIR = Path("datasets/20260913/mf_397_bf_gamma_clahe_um")        # 目標資料夾
TRAIN_RATIO = 0.8                     # 訓練集比例 (80%)
RANDOM_SEED = 42                      # 隨機種子 (確保結果可複現)
# 支持的圖片與標註檔案副檔名 (依需求調整)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_EXTS = {".txt", ".xml", ".json"}
# =======================================================


def split_and_validate_dataset():
    src_img_dir = SOURCE_DIR / "images" 
    src_lbl_dir = SOURCE_DIR / "labels"

    if not src_img_dir.exists() or not src_lbl_dir.exists():
        print(f"[錯誤] 找不到原始目錄：{src_img_dir} 或 {src_lbl_dir}")
        return

    # 1. 蒐集圖片與標註檔字典 {stem: Path}
    images_dict = {
        f.stem: f for f in src_img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS
    }
    labels_dict = {
        f.stem: f for f in src_lbl_dir.iterdir() if f.suffix.lower() in LABEL_EXTS
    }

    img_stems = set(images_dict.keys())
    lbl_stems = set(labels_dict.keys())

    # 2. 檢查對應狀態
    matched_stems = list(img_stems & lbl_stems)
    missing_label = img_stems - lbl_stems
    missing_image = lbl_stems - img_stems

    print("=== 資料集檢查報告 ===")
    print(f"原始圖片數量: {len(images_dict)}")
    print(f"原始標註數量: {len(labels_dict)}")
    print(f"成功配對對數: {len(matched_stems)}")

    if missing_label:
        print(f"⚠️  警告: 有 {len(missing_label)} 張圖片缺少對應標註檔 (已被排除)")
        # 若想查看詳細檔名，可取消下行註解:
        # print("缺少標註的圖片檔名:", sorted(list(missing_label)))

    if missing_image:
        print(f"⚠️  警告: 有 {len(missing_image)} 個標註檔缺少對應圖片 (已被排除)")

    if not matched_stems:
        print("[錯誤] 沒有任何成功配對的圖文組合，終止處理。")
        return

    # 3. 洗牌與拆分
    random.seed(RANDOM_SEED)
    random.shuffle(matched_stems)

    train_count = int(len(matched_stems) * TRAIN_RATIO)
    train_stems = matched_stems[:train_count]
    valid_stems = matched_stems[train_count:]

    print("\n=== 資料切分資訊 ===")
    print(f"Train 集數量: {len(train_stems)} (約 {TRAIN_RATIO*100:.0f}%)")
    print(f"Valid 集數量: {len(valid_stems)} (約 {(1-TRAIN_RATIO)*100:.0f}%)")

    # 4. 建立目標資料夾並複製檔案
    splits = {"train": train_stems, "valid": valid_stems}

    for split_name, stems in splits.items():
        dst_img_dir = TARGET_DIR / split_name / "images"
        dst_lbl_dir = TARGET_DIR / split_name / "labels"

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        for stem in stems:
            img_file = images_dict[stem]
            lbl_file = labels_dict[stem]

            # 複製檔案 (若要改用移動，可將 copy2 改為 move)
            shutil.copy2(img_file, dst_img_dir / img_file.name)
            shutil.copy2(lbl_file, dst_lbl_dir / lbl_file.name)

    print(f"\n✅ 處理完成！資料集已儲存至: {TARGET_DIR.resolve()}")


if __name__ == "__main__":
    split_and_validate_dataset()