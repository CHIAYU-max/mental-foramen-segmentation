from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 1. Load model
model = YOLO("runs/segment/20261008/yolov8n_seg_mf_559_unsharp_mask/weights/best.pt")

# 2. Run prediction (Note updated show_boxes argument)
results = model.predict(
    source="dental_project/20260919/test_um/images",
    conf=0.4,
    iou = 0.3,
    save=True,  # Automatically saves image predictions to runs/segment/predict/
    show_labels=True,
    show_conf=True,
    show_boxes=True,
)

# 3. View images inline or save rendered output
for result in results:
    # Plot array returns RGB format suitable for matplotlib
    annotated_img_rgb = result.plot(boxes=True, masks=True)

    # Option A: Display image using Matplotlib (Works well in Jupyter/GUI-enabled environments)
    plt.figure(figsize=(10, 10))
    plt.imshow(annotated_img_rgb)
    plt.axis("off")
    plt.title(Path(result.path).name)
    plt.show()