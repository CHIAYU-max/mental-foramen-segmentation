from ultralytics import YOLO

# 1. 載入你剛剛訓練最好的權重 (不要再用 yolov8s-seg.pt)
model = YOLO("./dental_project/train_success_replica3/weights/best.pt")

# 2. 開始訓練 (關鍵是降低 lr0)
model.train(
    data="yaml/20251217_alc_66.yaml",
    epochs=50,  # 不需要再跑 150 輪，微調 50 輪通常夠了
    batch=16,  # 維持原本設定
    imgsz=640,

    optimizer='AdamW',  # 繼續使用 AdamW
    lr0=0.0001,  # 【關鍵！】設為極低值 (原本的 1/100 或 1/10)
    # 讓模型在現有基礎上「輕輕地」修正，而不是劇烈改變

    # 保持原本有效的資料擴增
    scale=0.8,
    degrees=15.0,
    copy_paste=0.3
)