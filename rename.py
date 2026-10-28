from pathlib import Path


def batch_rename(
    folder_path: str,
    prefix: str = "doc_",
    replace_old: str = None,
    replace_new: str = None,
    dry_run: bool = True,
):
    """批次重新命名指定資料夾內的檔案

    :param folder_path: 目標資料夾路徑
    :param prefix: 新增的前綴 (若有流水號會接在後面)
    :param replace_old: 要被取代的字串 (選填)
    :param replace_new: 取代後的新字串 (選填)
    :param dry_run: 若為 True 僅預覽變更，確認無誤改為 False 才會真正執行
    """
    target_dir = Path(folder_path)

    if not target_dir.exists() or not target_dir.is_dir():
        print(f"錯誤：找不到路徑或該路徑不是資料夾 -> {folder_path}")
        return

    # 取得所有非隱藏檔案並排序
    files = sorted([f for f in target_dir.iterdir() if f.is_file()])

    if not files:
        print("資料夾內沒有檔案。")
        return

    print(
        f"--- 開始處理 (模式: {'預覽 (Dry Run)' if dry_run else '正式執行'}) ---"
    )

    for index, file in enumerate(files, start=1):
        extension = file.suffix  # 取得副檔名 (例如 .jpg, .pdf)
        stem = file.stem  # 取得不含副檔名的檔名

        # 模式 1：文字替換
        if replace_old is not None and replace_new is not None:
            new_name = stem.replace(replace_old, replace_new) + extension

        # 模式 2：前綴 + 3位數流水號 (例如 doc_001.pdf)
        else:
            new_name = f"{prefix}{index:03d}{extension}"

        new_path = target_dir / new_name

        # 避免覆蓋已存在的同名檔案
        if new_path.exists() and new_path != file:
            print(f"[略過] 目標檔名已存在：{new_name}")
            continue

        print(f"{file.name}  ->  {new_name}")

        if not dry_run:
            file.rename(new_path)

    print("--- 處理完成 ---")


# ================= 使用範例 =================
if __name__ == "__main__":
    # 1. 指定你的資料夾路徑 (支援相對路徑或絕對路徑)
    TARGET_FOLDER = r"dental_project/20260909/mf_297_original/labels"

    # 2. 測試執行：先用 dry_run=True 查看預覽結果
    # 範例 A：加前綴 + 流水號 (相片_001.jpg, 相片_002.jpg...)
    batch_rename(TARGET_FOLDER, prefix="first_", dry_run=False)

    # 範例 B：取代特定字串 (例如將 "draft_" 改為 "final_")
     #batch_rename(TARGET_FOLDER, replace_old="0827_", replace_new=" ", dry_run=True)

    # 確認預覽結果沒問題後，將 dry_run 設為 False 即可真正修改檔名