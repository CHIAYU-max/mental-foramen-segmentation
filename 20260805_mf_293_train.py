from ultralytics import YOLO

def main():
    model = YOLO("yolov8n-seg.pt") 

    results = model.train(
        data="yaml/20261008_mf_559.yaml",
        epochs=200,
        patience=50,            
        imgsz=1280,                
        batch=16,                 
        device=0,                  
        workers=8,
        project="20261008",
        name="yolov8n_seg_mf_559_unsharp_mask",
        save=True,
        plots=True,
    )

if __name__ == "__main__":
    main()