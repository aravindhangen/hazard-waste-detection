# Hazardous Waste Detection — Final Technical Report

Covers **Tasks 4–6** in depth (model selection, training, deployment). For the full academic scope see:

- [PROJECT_PROPOSAL.md](PROJECT_PROPOSAL.md) — Tasks 1 (proposal)
- [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md) — Task 1 (architecture + plan)
- [TASK_DELIVERABLES.md](TASK_DELIVERABLES.md) — Tasks 2–6 (deliverable checklist)

## 1. Executive Summary

This project implements an instance-segmentation system for detecting hazardous objects in scrap-yard imagery, focusing on **Cylinders** (explosive risk) and **Shock Absorbers** (toxic hydraulic-oil risk). After dataset preparation, leakage prevention, and experimental model comparison, **YOLOv5s-Seg (Run 4)** was selected as the production model and deployed through a FastAPI backend with a web dashboard for upload, webcam, and three-model comparison demos.

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
| **YOLOv5s-Seg** | Run 4 | Trained & evaluated — **production** |
| **YOLO11s-Seg** | Run 2 | Trained & evaluated |
| **YOLOv8s-Seg** | Run 3 | Trained & evaluated |

### Selection rationale

> Three YOLO-based instance-segmentation models — YOLOv5s-Seg, YOLO11s-Seg, and YOLOv8s-Seg — were experimentally evaluated using the same frozen dataset, train/validation/test split, image resolution, and held-out test set. YOLOv8s-Seg achieved the highest test mAP@0.50 (0.656) and Cylinder recall (0.720). YOLOv5s-Seg was selected as the **production** model because it provides a well-established academic baseline, smaller deployable weights (~15 MB), and strong cloud CPU compatibility while maintaining competitive Cylinder recall (0.710).

**Priority for comparison:** Cylinder Recall → mAP@0.50 → Recall → FPS

### Accuracy–speed trade-off

```text
                    ACCURACY
                       ↑
                       │
            YOLOv8 ●   │
                       │    ● YOLO11
                  YOLOv5 ●
                       │
                       └────────────────→ SPEED
```

YOLOv8s demonstrates the highest accuracy on the held-out test set. YOLOv5s offers the best balance of deployment practicality (CPU-friendly, compact weights) and safety-aligned Cylinder recall for the hosted demo.

---

## 6. Experimental Results (Held-Out Test Set — 39 Images)

### Three-model comparison summary

| Model | mAP@50 | Recall | Cylinder Recall | FPS | Role |
|-------|-------:|-------:|----------------:|----:|------|
| **YOLOv5s-Seg** | 0.589 | 0.599 | 0.710 | **63.3** | **Production** |
| YOLO11s-Seg | 0.643 | 0.648 | 0.714 | 26.6 | Tested |
| YOLOv8s-Seg | **0.656** | **0.663** | **0.720** | 68.7 | Tested (best accuracy) |

### YOLOv5s-Seg (Production — Run 4)

| Metric | Value |
|--------|------:|
| Precision | 0.653 |
| Recall | 0.599 |
| F1 | 0.625 |
| **mAP@0.50** | **0.589** |
| **mAP@0.50:0.95** | **0.361** |
| **Cylinder recall** | **0.710** |
| Shock Absorber recall | 0.487 |
| Inference (RTX 4050) | **~63.3 FPS** (15.8 ms) |

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

YOLOv8s achieved the highest mAP@0.50 and Cylinder recall on the frozen test set. YOLOv5s was chosen for production deployment because it is a standard academic baseline, runs efficiently on cloud CPU (Render Standard 2 GB), and keeps competitive Cylinder recall (71.0%). YOLO11s and YOLOv8s remain available in the dashboard for side-by-side comparison.

**Official artifacts:**

- `runs/yolov5s_run4/evaluation/run4_evaluation_report.json` (Run 4)
- `runs/yolov8s_run3/evaluation/run3_evaluation_report.json` (Run 3)
- `runs/yolo11s_run2/evaluation/run2_evaluation_report.json` (Run 2)
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
                    │   YOLOv5s-Seg     │
                    │ best_yolov5s.pt   │
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
        ───────────── COMPARISON ─────────────

                 Same Frozen Test Set
                         │
           ┌─────────────┼─────────────┐
           ▼             ▼             ▼
        YOLOv5s       YOLO11s       YOLOv8s
        0.589         0.643         0.656
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
| Production weights | `runs/yolov5s_run4/weights/best_yolov5s.pt` | Frozen Run 4 model |
| Run 3 weights | `runs/yolov8s_run3/weights/best_yolov8s.pt` | Comparison |
| Run 2 weights | `runs/yolo11s_run2/weights/best_yolo11s.pt` | Comparison |

---

## 8. Application Features

- Image upload with drag-and-drop
- Webcam capture and optional live scan (3 s interval)
- Confidence threshold slider
- Instance segmentation masks with class labels
- Hazard classification (explosive / toxic) and alerts
- Single-model inference (default: YOLOv5s)
- Side-by-side model comparison (YOLOv5s vs YOLO11s vs YOLOv8s)
- Benchmark table (experimentally trained models only)

**Endpoints:** `GET /health`, `GET /models`, `POST /predict`, `POST /predict/compare`

---

## 9. Limitations

1. **Small test set (39 images)** — metrics have statistical uncertainty; small deltas should not be over-interpreted.
2. **Domain-specific dataset** — performance may not generalize to all scrap-yard conditions, lighting, or camera angles.
3. **Two-class scope** — only Cylinder and Shock Absorber are supported.
4. **Cylinder recall 0.710** — still misses ~29% of cylinders on test; not sufficient for unsupervised autonomous operation without human oversight.
5. **Accuracy trade-off** — YOLOv8s outperforms YOLOv5s on test mAP@0.50; production choice favors deployability over peak accuracy.

---

## 10. Future Work

- Expand dataset (especially hard negatives and occluded cylinders)
- Edge deployment with speed-optimized models (e.g. YOLOv8s) where human oversight exists
- Active-learning loop for misclassified samples
- Multi-camera scrap-line integration

---

## 11. Conclusion

A complete ML prototype was delivered: dataset engineering → three-model experimentation → unbiased evaluation → deployment-oriented selection → API → dashboard → live inference. YOLOv5s-Seg (Run 4) serves as the production and academic baseline model, with YOLO11s and YOLOv8s available for benchmark comparison on the same frozen test split.
