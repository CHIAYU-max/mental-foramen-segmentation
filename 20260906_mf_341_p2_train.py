from dice_loss import CustomSegmentationTrainer


def main():
    trainer = CustomSegmentationTrainer(
        overrides={
            "model": "yaml/yolov8n-seg_p2.yaml",
            "data": "yaml/20260812_mf_293.yaml",
            "epochs": 150,
            "imgsz": 1280,
            "batch": 4,
            "mask_ratio": 2,
            "close_mosaic": 30,
            "scale": 0.15,
            "degrees": 5.0,
            "box": 5.0,
            "workers": 0,
        }
    )
    trainer.train()


if __name__ == "__main__":
    main()