import os
from ultralytics import YOLO

def run_segmentation(model_path, input_folder, output_folder, conf=0.25):
    """
    執行 YOLOv8-seg 預測並儲存結果
    
    Args:
        model_path (str): 模型權重路徑 (.pt)
        input_folder (str): 輸入圖片的資料夾路徑
        output_folder (str): 預測結果儲存的資料夾路徑
        conf (float): 置信度閾值
    """
    
    # 1. 載入模型
    model = YOLO(model_path)
    
    # 確保輸出目錄存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"建立輸出資料夾: {output_folder}")

    # 2. 執行預測
    # Ultralytics 會把結果儲存在 project/name；
    # 將 output_folder 拆成 project + name，才能精準對應到你想要的路徑。
    output_folder = os.path.abspath(os.path.normpath(output_folder))
    project_dir = os.path.dirname(output_folder) or "."
    run_name = os.path.basename(output_folder)

    # save=True 會自動將繪製好遮罩的圖片存檔
    results = model.predict(
        source=input_folder,
        conf=conf,
        save=True,          # 儲存預測後的圖片
        show_conf=False,    # 不在圖片上顯示信心分數
        show_labels=False,  # 不在圖片上顯示類別名稱
        project=project_dir,
        name=run_name,
        exist_ok=True,
        save_txt = True# 如果資料夾已存在則直接使用
    )

    actual_save_dir = str(results[0].save_dir) if results else output_folder
    print(f"\n預測完成！結果已儲存至: {actual_save_dir}")

if __name__ == "__main__":
    # --- 設定路徑 ---
    MODEL_PATH = 'dental_project/train_success_replica3/weights/best.pt'  # 也可以換成你訓練好的權重，如 'best.pt'
    INPUT_PATH = 'teeth_predict/original_images'   # 放圖片的資料夾
    OUTPUT_PATH = 'teeth_predict/predict_images' # 預測結果儲存路徑
    
    # 執行函數
    run_segmentation(MODEL_PATH, INPUT_PATH, OUTPUT_PATH)