# Hazardous Waste Detection — Project Proposal

## 1. Project Title

**Hazardous Waste Detection in Scrap Yards Using Instance Segmentation and Deep Learning**

Automated detection of **Cylinders** (explosive risk) and **Shock Absorbers** (toxic hydraulic-oil risk) in scrap-yard imagery using YOLO-based instance segmentation, FastAPI inference, and a web dashboard.

---

## 2. Problem Statement

Scrap yards and metal-recycling facilities process large volumes of mixed waste daily. Among ordinary scrap, **pressurized gas cylinders** and **automotive shock absorbers** are especially dangerous:

- **Cylinders** can explode when crushed or sheared by compactors.
- **Shock absorbers** may release toxic hydraulic fluid when ruptured.

Manual visual inspection is slow, inconsistent, and unsafe at scale. Workers cannot reliably identify partially buried, rusted, or occluded hazards under poor lighting. There is a need for an automated computer-vision system that can **detect, localize, and segment** hazardous objects in real time to support safer sorting decisions.

---

## 3. Project Objectives

| # | Objective |
|---|-----------|
| 1 | Collect and annotate scrap-yard images for two hazard classes: **Cylinder** and **Shock Absorber**. |
| 2 | Build a clean, balanced, and leakage-free dataset with train/validation/test splits. |
| 3 | Apply suitable data augmentation to improve model generalization. |
| 4 | Select and justify an appropriate deep-learning model for instance segmentation. |
| 5 | Train and evaluate the model using safety-aligned metrics (especially **Cylinder recall**). |
| 6 | Deploy the trained model via **FastAPI** with a dashboard for upload, webcam, and live inference. |

---

## 4. Proposed Solution

A end-to-end ML pipeline:

```text
Roboflow Annotation → Dataset Merge & QA → Augmentation & Balancing
        ↓
   YOLO Segmentation Training (275/79/39 split)
        ↓
   Held-out Test Evaluation (mAP, recall, per-class metrics)
        ↓
   FastAPI + Web Dashboard (upload / webcam / 3-model compare)
        ↓
   Hazard Alerts (Explosive / Toxic) for operators
```

**Key design choices:**

- **Instance segmentation** (pixel masks) rather than detection-only bounding boxes — provides precise object boundaries in cluttered scrap piles.
- **Two-class taxonomy** aligned with operational safety categories.
- **Frozen dataset** (`hazard_dataset_clean/`) so all model runs are comparable.
- **Production model:** YOLOv5s-Seg (Run 4); YOLO11s and YOLOv8s retained for benchmark comparison.

**Live demo:** [https://hazard-waste-detection.onrender.com/dashboard/](https://hazard-waste-detection.onrender.com/dashboard/)

---

## 5. Methodology / Approach

### Step-by-step plan (Tasks 1–6)

| Task | Phase | Activities | Deliverable |
|------|-------|------------|-------------|
| **1** | Planning | Literature review, problem scoping, architecture design | Project proposal + system architecture |
| **2** | Annotation | Roboflow labeling of cylinders and shock absorbers in scrap images | Annotated dataset (2 classes) |
| **3** | Augmentation | Roboflow + training-time augmentation; class-balance analysis | Augmented & balanced dataset |
| **4** | Model selection | Compare YOLO segmentation families; justify choice | Model selection report + dataset stats |
| **5** | Training | 70/20/10 split; train YOLOv5s/11s/8s; evaluate on test set | Evaluation scores & reports |
| **6** | Deployment | FastAPI `/predict`, health checks, dashboard, API tests | Deployed API + test output |

### Data pipeline (implemented)

```text
Task 2 — Roboflow exports (2 projects)
  • Shock Absorber.v1-v1.yolov9
  • scrap hazdetection.v1-v1.yolov9
        ↓
scripts/pipeline/00_build_dataset.py
  → hazard_dataset/  (512 images, 70/20/10)
        ↓
Task 3 — Dedup removes Roboflow duplicate augmentations
scripts/pipeline/05_deduplicate_and_resplit.py
  → hazard_dataset_clean/  (393 unique images)
        ↓
Task 5 — Frozen split: 275 train / 79 val / 39 test
```

### Evaluation methodology

- All models trained on the **same frozen split** at **640×640** resolution.
- Primary safety metric: **Cylinder recall** on held-out test set.
- Secondary metrics: mAP@0.50, mAP@0.50:0.95, precision, F1, inference FPS.

---

## 6. Tech Stack

| Layer | Technology |
|-------|------------|
| **Annotation** | [Roboflow](https://roboflow.com/) (polygon / instance labels, YOLO export) |
| **Language** | Python 3.10+ |
| **Deep learning** | PyTorch, Ultralytics YOLOv5 / YOLO11 / YOLOv8 (segmentation) |
| **Computer vision** | OpenCV, Pillow, `imagehash` (deduplication) |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | HTML/CSS/JavaScript dashboard |
| **Containerization** | Docker, Docker Compose |
| **Cloud deployment** | Render.com (CPU, Blueprint `render.yaml`) |
| **Version control** | Git, GitHub |
| **Testing** | `scripts/verify_endpoints.py` (httpx smoke tests) |

---

## 7. System Architecture

### High-level architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                              │
│  Web Browser  →  Dashboard (upload / webcam / compare UI)       │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP (JSON + multipart image)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        API LAYER (FastAPI)                       │
│  GET /health   GET /models   POST /predict   POST /predict/compare│
│  ModelManager  →  Yolov5Engine | UltralyticsEngine               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     INFERENCE LAYER                              │
│  YOLOv5s-Seg (production)  |  YOLO11s  |  YOLOv8s (compare)   │
│  Weights: runs/yolov5s_run4/weights/best_yolov5s.pt             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        OUTPUT LAYER                              │
│  Segmentation masks + class labels + confidence scores           │
│  Hazard classification: Cylinder → EXPLOSIVE | Shock → TOXIC    │
└─────────────────────────────────────────────────────────────────┘
```

### Repository layout

| Path | Role |
|------|------|
| `hazard_detection/` | Core config, paths, label utilities |
| `scripts/pipeline/` | Dataset build, dedup, validation (Tasks 2–4) |
| `scripts/training/` | Model training & comparison (Task 5) |
| `api/` | FastAPI application (Task 6) |
| `dashboard/` | Web UI |
| `hazard_dataset_clean/` | Frozen dataset manifest (`data.yaml`; images local-only) |
| `runs/yolov5s_run4/` | Production weights & evaluation reports |

See [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md) for detailed diagrams and module breakdown.

---

## 8. Expected Outcomes

| Outcome | Target / Achieved |
|---------|-------------------|
| Annotated dataset (2 classes) | ✅ 393 unique images after QA |
| Balanced classes | ✅ Instance ratio ~1.17:1 (Cylinder : Shock Absorber) |
| Train/val/test split | ✅ 275 / 79 / 39 (zero cross-split leakage) |
| Production model mAP@0.50 (test) | ✅ **0.589** (YOLOv5s-Seg) |
| Cylinder recall (test) | ✅ **0.710** (YOLOv5s-Seg) |
| Real-time inference (GPU) | ✅ ~63 FPS (RTX 4050, YOLOv5s) |
| Deployed API | ✅ FastAPI on localhost + Render cloud |
| Interactive dashboard | ✅ Single-model + 3-model compare |

---

## 9. Evaluation Metrics

### Segmentation metrics (held-out test set — 39 images)

| Metric | Description | YOLOv5s (Production) |
|--------|-------------|---------------------:|
| **Precision** | Fraction of predictions that are correct | 0.653 |
| **Recall** | Fraction of ground-truth objects found | 0.599 |
| **F1** | Harmonic mean of precision and recall | 0.625 |
| **mAP@0.50** | Mean average precision at IoU 0.50 (masks) | **0.589** |
| **mAP@0.50:0.95** | COCO-style mAP across IoU thresholds | 0.361 |
| **Cylinder recall** | Recall for class 0 (safety-critical) | **0.710** |
| **Shock Absorber recall** | Recall for class 1 | 0.487 |
| **FPS** | Inference speed (RTX 4050) | 63.3 |

### Three-model comparison (same test set)

| Model | mAP@50 | Cylinder Recall | FPS | Role |
|-------|-------:|----------------:|----:|------|
| YOLOv5s-Seg | 0.589 | 0.710 | 63.3 | **Production** |
| YOLO11s-Seg | 0.643 | 0.714 | 26.6 | Benchmark |
| YOLOv8s-Seg | **0.656** | **0.720** | 68.7 | Benchmark (best accuracy) |

**Official reports:** `runs/yolov5s_run4/evaluation/`, `runs/comparison/all_runs_comparison.txt`

---

## 10. Future Enhancements

1. **Expand dataset** — more occluded cylinders, hard negatives, multi-angle scrap-line cameras.
2. **Improve Shock Absorber recall** — currently 0.487 on test; targeted collection and rebalancing.
3. **Edge deployment** — ONNX/TensorRT export for on-premise GPU or Jetson devices.
4. **Active learning** — flag low-confidence predictions for human re-labeling in Roboflow.
5. **Multi-camera integration** — conveyor-belt monitoring with alert webhooks.
6. **Model upgrade path** — swap production weights to YOLOv8s where GPU is available without changing API contract.

---

## 11. Conclusion

This project delivers a complete hazardous-waste detection prototype spanning **data annotation**, **augmentation and balancing**, **model selection**, **training and evaluation**, and **FastAPI deployment**. Using Roboflow, we labeled cylinders and shock absorbers in scrap-yard imagery and built a leakage-free dataset of 393 unique images. After experimental comparison of three YOLO segmentation models on a frozen test split, **YOLOv5s-Seg (Run 4)** was selected as the production model for its academic baseline role, compact weights (~15 MB), and strong CPU deployability while maintaining **71.0% Cylinder recall**. The system is accessible through a FastAPI backend, interactive web dashboard, and a public Render deployment, demonstrating real-time hazard alerts for operators in scrap-recycling environments.

---

## Related documents

| Document | Content |
|----------|---------|
| [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md) | Detailed architecture + step-by-step plan (Task 1) |
| [TASK_DELIVERABLES.md](TASK_DELIVERABLES.md) | Task 2–6 deliverables with artifact paths |
| [FINAL_REPORT.md](FINAL_REPORT.md) | Technical report (Tasks 4–5 focus) |
| [PRESENTATION.md](PRESENTATION.md) | Slide deck outline |
