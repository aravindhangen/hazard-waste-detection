# Academic Task Deliverables (Tasks 1–6)

Mapping of project coursework requirements to repository artifacts and scripts.

---

## Task 1 — Project Architecture & Proposal

| Deliverable | Location |
|-------------|----------|
| **Project Architecture** | [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md) |
| **Project Proposal** | [PROJECT_PROPOSAL.md](PROJECT_PROPOSAL.md) |
| **Step-by-step plan** | [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md) §4 |
| **Presentation outline** | [PRESENTATION.md](PRESENTATION.md) |

---

## Task 2 — Data Annotation (Roboflow)

### Objective

Annotate hazardous objects in scrap-yard images using Roboflow. Two categories: **Cylinder** and **Shock Absorber**.

### Deliverables

| # | Deliverable | Status | Evidence |
|---|-------------|--------|----------|
| 1 | **Data annotations** (2 classes) | ✅ Complete | `hazard_detection/config/dataset.py` — `CLASS_NAMES` |
| 2 | **Annotated dataset** | ✅ Complete | Merged into `hazard_dataset/` then cleaned to `hazard_dataset_clean/` |

### Annotation details

| Item | Value |
|------|-------|
| Tool | [Roboflow](https://roboflow.com/) |
| Source project 1 | `Shock Absorber.v1-v1.yolov9` (prefix: `shock_`) |
| Source project 2 | `scrap hazdetection.v1-v1.yolov9` (prefix: `scrap_`) |
| Classes | `0: Cylinder`, `1: Shock_Absorber` |
| Label format | YOLO polygon segmentation (normalized coordinates) |
| Bounding boxes | Implicit from polygon vertices (min/max envelope) |

> **Note:** Roboflow exports polygon segmentation labels. Each `.txt` line contains `class_id x1 y1 x2 y2 …` normalized polygon points. Axis-aligned bounding boxes can be derived from these polygons for detection-only use cases.

### Pipeline scripts

| Script | Purpose |
|--------|---------|
| `scripts/pipeline/00_build_dataset.py` | Merge two Roboflow exports → `hazard_dataset/` |
| `scripts/pipeline/01_dataset_audit.py` | Count images, labels, class distribution |
| `scripts/pipeline/02_validate_labels.py` | Validate polygon format |
| `scripts/pipeline/04_visualize_annotations.py` | Overlay masks on images for QA |

### Dataset manifest (in Git)

- `hazard_dataset/data.yaml`
- `hazard_dataset_clean/data.yaml`

Images and label files are **local-only** (not published to GitHub for privacy). Restore from Roboflow backup or re-run the pipeline scripts.

---

## Task 3 — Data Augmentation & Balancing

### Objective

Apply augmentation to increase diversity and ensure balanced representation of both classes.

### Deliverables

| # | Deliverable | Status | Evidence |
|---|-------------|--------|----------|
| 1 | **Augmented dataset** | ✅ Complete | 512 raw images (Roboflow + offline aug) → 393 unique after dedup |
| 2 | **Balanced dataset** | ✅ Complete | Class instance ratio **1.17:1** — no oversampling required |

### Augmentation techniques applied

#### A. Roboflow / offline augmentation (export time)

Roboflow generates augmented copies of source images (visible as duplicate screenshots with different `rf.*` hash suffixes). Techniques typically include:

- Horizontal flip
- Rotation / skew
- Brightness / exposure variation
- Crop and resize

These contributed to the initial **512-image** pool before deduplication.

#### B. Deduplication (quality step)

`scripts/pipeline/05_deduplicate_and_resplit.py`:

- Perceptual hashing (`imagehash.phash`) removes near-duplicate augmented copies
- **119 images removed** → **393 unique** images
- **0 cross-split leakage** after stratified resplit

Report: `reports/duplicate_removal_report.txt`

#### C. Training-time augmentation (YOLO)

Applied automatically during model training:

| Technique | Parameter |
|-----------|-----------|
| Mosaic | 1.0 |
| Horizontal flip | 0.5 |
| HSV hue / sat / val | 0.015 / 0.7 / 0.4 |
| Translate | 0.1 |
| Scale | 0.5 |
| RandAugment | enabled |
| Close mosaic (last N epochs) | 10 |

### Class balance analysis

| Metric | Value |
|--------|------:|
| Cylinder instances | 1,167 |
| Shock Absorber instances | 997 |
| Ratio | **1.17 : 1** |
| Decision | **Balanced** — no SMOTE / oversampling applied |

Validation output: `scripts/pipeline/06_final_dataset_validation.py`

---

## Task 4 — Model Selection & Dataset Preparation

### Deliverables

| # | Deliverable | Status | Evidence |
|---|-------------|--------|----------|
| 1 | **Model selection** with justification | ✅ Complete | [PROJECT_PROPOSAL.md](PROJECT_PROPOSAL.md) §5, [FINAL_REPORT.md](FINAL_REPORT.md) §5 |
| 2 | **Combined dataset** (Cylinder + Shock Absorber) | ✅ Complete | `hazard_dataset_clean/` |
| 3 | **Class distribution analysis** | ✅ Complete | `01_dataset_audit.py`, dedup report |
| 4 | **Balance assessment** | ✅ Balanced (1.17:1) | See Task 3 above |

### Models considered

| Model | Reason considered | Benchmarked |
|-------|-------------------|:-----------:|
| Mask R-CNN | Strong instance segmentation baseline | No (compute cost) |
| RT-DETR | Real-time transformer detector | No |
| **YOLOv5s-Seg** | Academic baseline, compact, CPU-friendly | ✅ Run 4 |
| **YOLO11s-Seg** | Modern Ultralytics architecture | ✅ Run 2 |
| **YOLOv8s-Seg** | Best accuracy in experiments | ✅ Run 3 |

### Why YOLOv5s-Seg was chosen (production)

1. **Instance segmentation** — pixel masks for cluttered scrap scenes.
2. **Safety-aligned recall** — 71.0% Cylinder recall on held-out test (vs 72.0% for YOLOv8s).
3. **Deployment** — ~15 MB weights, runs on Render CPU Standard tier.
4. **Academic baseline** — well-cited, reproducible training via official `ultralytics/yolov5` repo.

YOLOv8s remains the **accuracy leader** and is available in the dashboard for comparison.

---

## Task 5 — Training & Evaluation

### Deliverables

| # | Deliverable | Status | Evidence |
|---|-------------|--------|----------|
| 1 | **Dataset split** (train / val / test) | ✅ Complete | 275 / 79 / 39 |
| 2 | **Trained model** | ✅ Complete | `runs/yolov5s_run4/weights/best_yolov5s.pt` |
| 3 | **Evaluation scores** | ✅ Complete | See tables below |

### Dataset split

| Split | Images | Ratio |
|-------|-------:|------:|
| Train | 275 | 70% |
| Validation | 79 | 20% |
| Test | 39 | 10% |
| **Total** | **393** | 100% |

Split is **frozen** in `hazard_dataset_clean/`. Random seed: `42`.

### Training configuration (Run 4 — production)

| Parameter | Value |
|-----------|-------|
| Model | YOLOv5s-Seg |
| Pretrained | `yolov5s-seg.pt` |
| Epochs | 100 |
| Image size | 640 × 640 |
| Batch size | 4 |
| Optimizer | SGD (YOLOv5 default) |
| Script | `scripts/training/13_train_yolov5_run4.py` |

### Evaluation scores — YOLOv5s-Seg (test set)

| Metric | Score |
|--------|------:|
| Precision | 0.653 |
| Recall | 0.599 |
| F1 | 0.625 |
| **mAP@0.50** | **0.589** |
| mAP@0.50:0.95 | 0.361 |
| **Cylinder recall** | **0.710** |
| Shock Absorber recall | 0.487 |
| Inference (RTX 4050) | 63.3 FPS |

### Per-class test metrics (masks)

| Class | Precision | Recall | F1 | mAP@50 |
|-------|----------:|-------:|---:|-------:|
| Cylinder | 0.624 | **0.710** | 0.664 | 0.617 |
| Shock_Absorber | 0.681 | 0.487 | 0.568 | 0.560 |

### Official report files

| File | Content |
|------|---------|
| `runs/yolov5s_run4/evaluation/run4_evaluation_report.txt` | Run 4 human-readable report |
| `runs/yolov5s_run4/evaluation/run4_evaluation_report.json` | Run 4 machine-readable metrics |
| `runs/yolo11s_run2/evaluation/run2_evaluation_report.json` | Run 2 metrics |
| `runs/yolov8s_run3/evaluation/run3_evaluation_report.json` | Run 3 metrics |
| `runs/comparison/all_runs_comparison.txt` | 3-model summary |
| `runs/comparison/all_runs_comparison.json` | 3-model JSON |

### Training scripts

```bash
python scripts/training/13_train_yolov5_run4.py    # Production (Run 4)
python scripts/training/09_train_yolo11_run2.py    # Benchmark (Run 2)
python scripts/training/11_train_yolov8_run3.py    # Benchmark (Run 3)
python scripts/training/12_compare_all_runs.py     # Comparison report
```

---

## Task 6 — FastAPI Deployment & API Testing

### Deliverables

| # | Deliverable | Status | Evidence |
|---|-------------|--------|----------|
| 1 | **FastAPI application** | ✅ Complete | `api/main.py` (not `app.py` — modular package layout) |
| 2 | **Prediction API endpoint** | ✅ Complete | `POST /predict`, `POST /predict/compare` |
| 3 | **API testing** | ✅ Complete | `scripts/verify_endpoints.py` |

### API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service health + model load status |
| `GET` | `/models` | Available model catalog + benchmarks |
| `POST` | `/predict` | Single-image inference |
| `POST` | `/predict/compare` | Multi-model comparison |
| `GET` | `/docs` | Swagger UI (interactive API docs) |
| `GET` | `/dashboard/` | Web UI |

### Example: start server

```bash
# Windows
run_server.bat

# Or manually
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Example: test prediction (curl)

```bash
curl -X POST "http://127.0.0.1:8000/predict?model_id=yolov5&include_annotated_image=false" \
  -F "file=@demo/01_clear_cylinder.jpg"
```

### Example: automated smoke test

```bash
python scripts/verify_endpoints.py
```

Expected output:

```text
[OK] GET /health -> 200
[OK] GET /models -> 200
[OK] POST /predict -> 200
...
```

### Live deployment

| Environment | URL |
|-------------|-----|
| Local | http://127.0.0.1:8000/dashboard/ |
| Render (cloud) | https://hazard-waste-detection.onrender.com/dashboard/ |

Deployment guide: [RENDER.md](RENDER.md)

### Sample API response fields

```json
{
  "hazard_detected": true,
  "hazard_type": "explosive",
  "detections": [
    {
      "class_name": "Cylinder",
      "confidence": 0.82,
      "bbox": [x1, y1, x2, y2]
    }
  ],
  "inference_ms": 15.8,
  "model_id": "yolov5"
}
```

---

## Quick reference — all documentation

| Document | Tasks covered |
|----------|---------------|
| [PROJECT_PROPOSAL.md](PROJECT_PROPOSAL.md) | 1 (proposal sections 1–11) |
| [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md) | 1 (architecture + plan) |
| [TASK_DELIVERABLES.md](TASK_DELIVERABLES.md) | 2, 3, 4, 5, 6 (this file) |
| [FINAL_REPORT.md](FINAL_REPORT.md) | 4, 5 (technical deep-dive) |
| [DEMO_SCRIPT.md](DEMO_SCRIPT.md) | 6 (live demo rehearsal) |
| [RENDER.md](RENDER.md) | 6 (cloud deploy) |
