import os
from ultralytics import YOLO

def main():
    # ==========================================
    # 1. 路徑設定
    # ==========================================
    # 填入你訓練出來最強大的 best.pt 路徑
    model_path = "./runs/pose/yolov11-pose/run1-5/weights/best.pt" 
    
    # 填入那 200 多張「尚未標註」的圖片資料夾路徑 (你想看結果的圖片)
    test_images_dir = "./original_dataset/images" 
    
    # ==========================================
    # 2. 載入模型
    # ==========================================
    print(f"正在載入模型: {model_path}...")
    model = YOLO(model_path)

    print(f"開始預測並繪製視覺化結果... 圖片來源: {test_images_dir}")
    
    # ==========================================
    # 3. 執行推論與自動繪圖 (核心設定)
    # ==========================================
    # save=True: 這是魔法參數，會自動在圖片上畫框跟點並儲存
    # conf: 信心門檻，0.5 代表只畫出模型有 50% 以上把握的目標
    # line_width: 控制畫框跟畫點的粗細，你可以依喜好調整 (預設通常是 3)
    results = model.predict(
        source=test_images_dir,
        conf=0.5,
        imgsz=1024,
        save=True,          # 啟動自動存圖功能
        stream=True,        # 適合大量圖片，不會塞爆記憶體
        show_labels=False,  # 設為 False 隱藏 "tooth" 文字，讓畫面更乾淨只看點跟框
        show_conf=False,    # 設為 False 隱藏機率數字
        line_width=2        # 線條粗細 (調細一點比較容易看清楚精準的 ALC 點)
    )

    # 由於我們啟用了 stream=True，它會變成一個產生器 (Generator)
    # 我們必須寫一個空的 for 迴圈來推動它把所有圖片跑完
    count = 0
    save_dir = "./runs/predict"
    for r in results:
        count += 1
        if count == 1:
            save_dir = r.save_dir # 記錄 YOLO 自動產生存檔路徑
            
    print("==========================================")
    print(f"✅ 視覺化預測完成！共處理 {count} 張圖片。")
    print(f"👉 請前往資料夾查看結果圖： {save_dir}")

if __name__ == "__main__":
    main()