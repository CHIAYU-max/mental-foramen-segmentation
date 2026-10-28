# Mental foramen and inferior alveolar canal segmentation on panoramic radiographs

Before implant surgery in the lower premolar region, the surgeon has to avoid the mental foramen (the nerve exit)
and the inferior alveolar canal. This project finds both automatically on dental panoramic radiographs, in
collaboration with dentists at Linkou Chang Gung Memorial Hospital.

The radiographs and annotations are clinical data and are **not** included. This repository holds the code only.

## Pipeline

```
panoramic radiograph
 └─ stage 1: YOLOv8s-seg tooth instance segmentation (imgsz 1024)                 fdi_cropped.py
     └─ filtering → upper / lower jaw split (k-means) → FDI tooth numbering
         └─ left and right ROI around teeth 34–36 / 44–46, down to the mandible's lower border
             └─ unsharp mask                                                        mf_unsharp_mask.py
                 └─ stage 2: segment the mental foramen (mf) and the canal (ian)
                     ├─ U-Net with a ResNet34 encoder, 512 × 512                   kfold_unet_variants.py
                     └─ YOLO-seg (v8 / 11 / 26, n / s, imgsz 1280)
                         └─ foramen centroid, hit rate, tooth position (Tebo & Telford)  mf_position.py
```

## Results

559 ROIs cropped from at least 294 panoramic radiographs, 5-fold cross-validation grouped by patient and source
radiograph (`kfold_pano.py`), plus an independent test set of 137 ROIs. A foramen counts as found ("hit") when the
predicted centroid lies within the ground-truth foramen's equivalent radius. Five training seeds, test set with a
five-fold pixel-wise majority vote:

| Model | Test hit rate | Test mf Dice | Test ian Dice |
| --- | --- | --- | --- |
| U-Net (ResNet34) | 92.6 ± 1.7% | 0.71 | 0.68 |
| YOLO26s-seg (best of six YOLO models) | 87.7 ± 1.7% | 0.71 | 0.68 |

- With the five seeds pooled and a patient-level bootstrap, the U-Net finds more foramina: +4.8 points on the test
  set (95% CI 1.6–8.2, p = 0.006) and +1.9 points in cross-validation (95% CI 0.2–3.6, p = 0.034). Dice is not
  significantly different (`pano_stats.py`, `seed_bootstrap.py`).
- Accuracy is limited by how visible the foramen is: on images graded as visible, Dice reaches 0.84, above the
  annotator's own repeat agreement (0.79) (`pano_visibility.py`).
- Disagreement among models trained with different seeds flags uncertain cases without labels: referring the 10%
  most uncertain ROIs to a clinician raises the hit rate on the rest from 92.7% to 97.6% on the test set
  (`pano_uncertainty.py`).
- Deployment: per-layer rooflines and end-to-end latency (0.15 s per radiograph on a GPU, 0.4–1.1 s without one;
  `roofline_profile.py`, `pipeline_timing.py`), INT8 quantization with OpenVINO NNCF (2.5–3× faster on a CPU;
  `int8_quant.py`, `int8_ensemble.py`), and a cascade that stops after two folds when they agree (49% less compute
  at the same hit rate; `cascade_infer.py`).

## Main scripts

| Script | Purpose |
| --- | --- |
| `fdi_cropped.py` | Stage 1: tooth segmentation, filtering, FDI numbering, ROI cropping |
| `kfold_pano.py` | Stage 2 model comparison: grouped 5-fold split, U-Net and six YOLO models, evaluation |
| `kfold_unet_variants.py` | U-Net model (`ResNetUNet`), configurations, training and inference |
| `paper_metrics.py` | Metrics: Dice, hit rate, cross-validation and test-set evaluation |
| `pano_stats.py`, `seed_bootstrap.py` | Paired tests and patient-level bootstrap across seeds |
| `pano_visibility.py`, `pano_uncertainty.py` | Visibility-stratified results; model disagreement as an uncertainty score |
| `stage1_eval.py`, `stage1_failures.py`, `e2e_eval.py` | Stage-1 validation and the end-to-end comparison with manual ROIs |
| `mf_position.py`, `mf_position_eval.py`, `position_server.py` | Tooth-position rule, its evaluation, and a browser labeling tool |
| `roofline_profile.py`, `pipeline_timing.py`, `int8_quant.py` | Performance analysis and INT8 quantization |
| `roofline_workload.py`, `accel_model.py` | Workload characterization and a first-order analytical model of an INT8 accelerator |

Scripts run from the repository root and import shared constants from each other (`compare_yolo_unet.py`,
`kfold_yolo_unet.py`, `kfold_unet_variants.py`, `kfold_patient.py`). Data paths point to a `datasets/` folder that is
not part of this repository. Many older scripts from earlier stages of the project are kept as they were.

Environment: Python 3.10, PyTorch 2.9, Ultralytics 8.4, OpenVINO 2026.4 with NNCF.

## Credits

Research with the Biomedical/Multimedia IC and System Design Lab, Chung Yuan Christian University (advisor:
Prof. Shih-Lun Chen), and dentists at Linkou Chang Gung Memorial Hospital. Many of the evaluation, statistics and
performance-analysis scripts were written with the help of an AI assistant (Claude).
