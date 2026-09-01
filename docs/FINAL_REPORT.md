# Hazardous Waste Detection — Final Report (Tasks 4 & 5)

## 1. Executive Summary

This project implements an instance-segmentation system for detecting hazardous objects in scrap-yard imagery, focusing on **Cylinders** (explosive risk) and **Shock Absorbers** (toxic hydraulic-oil risk). After dataset preparation, leakage prevention, and experimental model comparison, **YOLOv9 GELAN-C-SEG** was selected as the production model and deployed through a FastAPI backend with a web dashboard for upload, webcam, and model-comparison demos.

---

## 2. Problem Statement

Scrap yards contain hazardous items that are difficult to identify visually at scale. Pressurized cylinders and shock absorbers pose safety risks if mishandled during sorting or compression. Manual inspection is slow and inconsistent. An automated computer-vision system is needed to detect and segment these objects in real time.

---

## 3. Objectives

1. Build a high-quality, leakage-free dataset for instance segmentation.
2. Select and train a segmentation model optimized for hazardous-object detection.
3. Evaluate models on a held-out test set with safety-aligned metrics.
4. Deploy inference through an API and interactive dashboard.

---

## 4. Dataset

| Item | Value |
|------|------:|
| Raw merged images | 512 |
| After deduplication & QA | **393 unique images** |
| Train / Val / Test split | **275 / 79 / 39** |
| Classes | Cylinder, Shock Absorber |
| Class instance ratio | 1,167 : 997 (1.17:1) |
| Cross-split duplicate leakage | **0** |
| Malformed polygon labels (final set) | **0** |

### Data pipeline

```text
512 raw images (two Roboflow exports)
        ↓
Deduplication (perceptual hash) + label cleaning
        ↓
393 unique images
        ↓
Stratified split: 275 train / 79 val / 39 test
        ↓
hazard_dataset_clean/  (frozen)
```

### QA measures

- Perceptual-hash duplicate detection and removal
- Polygon validation (minimum 3 points, normalized coordinates)
- Image–label pairing checks
- Stratified resplit preserving class balance
- Visual QA exports in `visual_qa/`

---

## 5. Model Selection

### Theoretical candidates considered

- Mask R-CNN
- YOLOv5-Seg
- YOLOv8-Seg
- RT-DETR

### Experimentally benchmarked (same frozen dataset, 275/79/39 split, 640×640, held-out test set)

| Model | Run | Status |
|-------|-----|--------|
| **YOLOv8s-Seg** | Run 3 | Trained & evaluated |
| **YOLOv9 GELAN-C-SEG** | Run 1 | Trained & evaluated |
| **YOLO11s-Seg** | Run 2 | Trained & evaluated |

### Selection rationale

> Three YOLO-based instance-segmentation models — YOLOv8s-Seg, YOLOv9 GELAN-C-SEG, and YOLO11s-Seg — were experimentally evaluated using the same frozen dataset, train/validation/test split, image resolution, and held-out test set. YOLOv9 GELAN-C-SEG achieved the highest test mAP@0.50 (0.732), overall recall (0.722), and Cylinder recall (0.734). Although YOLOv8s-Seg achieved the highest inference speed (68.7 FPS), its detection performance was lower. Since reliable detection of hazardous cylinders is the primary safety requirement, YOLOv9 was selected as the production model.

**Priority for comparison:** Cylinder Recall → mAP@0.50 → Recall → FPS

### Accuracy–speed trade-off

```text
                    ACCURACY
                       ↑
                       │
                  YOLOv9 ●
                       │
            YOLOv8 ●   │    ● YOLO11
                       │
                       └────────────────→ SPEED
```

YOLOv8s demonstrates a clear speed advantage (68.7 FPS, above the original 30 FPS design target) but lower safety-aligned detection metrics. YOLOv9 was selected for reliability, not maximum throughput.

---

## 6. Experimental Results (Held-Out Test Set — 39 Images)

### Three-model comparison summary

| Model | mAP@50 | Recall | Cylinder Recall | FPS | Decision |
|-------|-------:|-------:|----------------:|----:|----------|
| YOLOv8s-Seg | 0.656 | 0.663 | 0.720 | **68.7** | Not selected |
| **YOLOv9 GELAN-C-SEG** | **0.732** | **0.722** | **0.734** | 22.9 | **Selected** |
| YOLO11s-Seg | 0.643 | 0.648 | 0.714 | 26.6 | Not selected |

### YOLOv9 GELAN-C-SEG (Production — Run 1)

| Metric | Value |
|--------|------:|
| Precision | 0.743 |
| Recall | 0.722 |
| F1 | 0.732 |
| **mAP@0.50** | **0.732** |
| **mAP@0.50:0.95** | **0.510** |
| **Cylinder recall** | **0.734** |
| Shock Absorber recall | 0.709 |
| Inference (RTX 4050) | **~22.9 FPS** (43.6 ms) |

### YOLOv8s-Seg (Run 3)

| Metric | Value |
|--------|------:|
| Precision | 0.699 |
| Recall | 0.663 |
| F1 | 0.681 |
| mAP@0.50 | 0.656 |
| mAP@0.50:0.95 | 0.450 |
| Cylinder recall | 0.720 |
| Inference (RTX 4050) | **~68.7 FPS** (14.6 ms) |

### YOLO11s-Seg (Run 2)

| Metric | Value |
|--------|------:|
| Precision | 0.702 |
| Recall | 0.648 |
| F1 | 0.674 |
| mAP@0.50 | 0.643 |
| mAP@0.50:0.95 | 0.444 |
| Cylinder recall | 0.714 |
| Inference (RTX 4050) | **~26.6 FPS** (37.7 ms) |

### Decision

YOLOv9 achieved the highest mAP@0.50, overall recall, and Cylinder recall. YOLOv8s is dramatically faster but has lower Cylinder recall (72.0% vs 73.4%). YOLO11s is faster than YOLOv9 but lower on all detection metrics. For hazardous-waste detection, missing a cylinder is more important than gaining FPS — YOLOv9 remains production.

**Official artifacts:**

- `evaluation_reports/final_evaluation_report.txt` (Run 1)
- `runs/yolov8s_run3/evaluation/run3_evaluation_report.txt` (Run 3)
- `runs/yolo11s_run2/evaluation/run2_evaluation_report.txt` (Run 2)
- `runs/comparison/all_runs_comparison.txt`

---

## 7. System Architecture

### Production inference path

```text
                    ┌───────────────────┐
                    │   User / Operator │
                    └─────────┬─────────┘
                              │
                    Image / Webcam
                              │
                              ▼
                    ┌───────────────────┐
                    │  Web Dashboard    │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │     FastAPI       │
                    │     /predict      │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ YOLOv9 GELAN-C-SEG│
                    │    best.pt        │
                    │   PRODUCTION      │
                    └─────────┬─────────┘
                              │
                     Masks + Classes
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
                Cylinder          Shock Absorber
                EXPLOSIVE              TOXIC
                    │                   │
                    └─────────┬─────────┘
                              ▼
                       Hazard Alert
                              │
                              ▼
                         Dashboard
```

### Experimental comparison path

```text
        ───────────── EXPERIMENTAL ─────────────

                 Same Frozen Test Set
                         │
           ┌─────────────┼─────────────┐
           ▼             ▼             ▼
        YOLOv8s       YOLOv9       YOLO11s
        0.656         0.732         0.643
       mAP@50        mAP@50        mAP@50
           │             │             │
           └─────────────┼─────────────┘
                         ▼
                  Model Comparison
                  (Dashboard + API)
```

### Components

| Component | Path | Role |
|-----------|------|------|
| Core library | `hazard_detection/` | Config, paths, label utilities |
| API | `api/` | FastAPI, model manager, inference |
| Dashboard | `dashboard/` | Upload, webcam, comparison UI |
| Production weights | `yolov9/.../best.pt` | Frozen Run 1 model |
| Run 3 weights | `runs/yolov8s_run3/weights/best_yolov8s.pt` | Experimental |
| Run 2 weights | `runs/yolo11s_run2/weights/best_yolo11s.pt` | Experimental |

---

## 8. Application Features

- Image upload with drag-and-drop
- Webcam capture and optional live scan (3 s interval)
- Confidence threshold slider
- Instance segmentation masks with class labels
- Hazard classification (explosive / toxic) and alerts
- Single-model inference (default: YOLOv9)
- Side-by-side model comparison (YOLOv9 vs YOLO11s vs YOLOv8s)
- Benchmark table (experimentally trained models only)

**Endpoints:** `GET /health`, `GET /models`, `POST /predict`, `POST /predict/compare`

---

## 9. Limitations

1. **Small test set (39 images)** — metrics have statistical uncertainty; small deltas should not be over-interpreted.
2. **Domain-specific dataset** — performance may not generalize to all scrap-yard conditions, lighting, or camera angles.
3. **Two-class scope** — only Cylinder and Shock Absorber are supported.
4. **Cylinder recall 0.734** — still misses ~26% of cylinders on test; not sufficient for unsupervised autonomous operation without human oversight.
5. **Measured FPS below design target** — original target was 30+ FPS; YOLOv9 measured ~22.9 FPS on RTX 4050 (YOLOv8s reached 68.7 FPS, illustrating the accuracy–speed trade-off).

---

## 10. Future Work

- Expand dataset (especially hard negatives and occluded cylinders)
- Edge deployment with speed-optimized models (e.g. YOLOv8s) where human oversight exists
- Active-learning loop for misclassified samples
- Multi-camera scrap-line integration

---

## 11. Conclusion

A complete ML prototype was delivered: dataset engineering → three-model experimentation → unbiased evaluation → safety-oriented selection → API → dashboard → live inference. YOLOv9 GELAN-C-SEG was selected based on measured evidence (highest mAP@0.50 0.732 and Cylinder recall 0.734 among YOLOv8s, YOLOv9, and YOLO11s) rather than model novelty or speed alone.
