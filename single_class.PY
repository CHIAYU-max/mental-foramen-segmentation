import shutil
from pathlib import Path

# 原始與目標資料夾路徑
project_dir = Path(__file__).resolve().parent
src_dataset_dir = project_dir / "datasets/20260904/20260904_mf_456_unsharp_mask_0.5__brightness_0.2"
dst_dataset_dir = project_dir / "datasets/20260905/20260905_mf_456_unsharp_mask_0.5__brightness_0.2_single_class"

# 複製影像，並只保留 class 0 (mf) 的標註行
for split in ["train", "valid"]:
    src_split = src_dataset_dir / split
    dst_split = dst_dataset_dir / split
    src_images_dir = src_split / "images"
    src_labels_dir = src_split / "labels"
    dst_images_dir = dst_split / "images"
    dst_labels_dir = dst_split / "labels"

    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    for image_file in src_images_dir.iterdir():
        if image_file.is_file():
            shutil.copy2(image_file, dst_images_dir / image_file.name)

    for txt_file in src_labels_dir.glob("*.txt"):
        new_lines = []
        with txt_file.open("r", encoding="utf-8") as source_file:
            for line in source_file:
                fields = line.split()
                if fields and fields[0] == "0":
                    new_lines.append(line)

        # 寫入新的單類別標註檔（原標註檔完全不受影響）
        with (dst_labels_dir / txt_file.name).open("w", encoding="utf-8") as target_file:
            target_file.writelines(new_lines)

print("單一類別 Labels 生成完成！")