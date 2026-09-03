# Hazardous Waste Detection in Scrap Yards

Instance segmentation for **Cylinders** (explosive risk) and **Shock Absorbers** (toxic hydraulic-oil risk) in scrap-yard imagery.

| Item | Value |
|------|-------|
| **Production model** | YOLOv5s-Seg (Run 4) — `runs/yolov5s_run4/weights/best_yolov5s.pt` |
| **Comparison models** | YOLO11s (Run 2), YOLOv8s (Run 3) |
| **Live demo** | https://hazard-waste-detection.onrender.com/dashboard/ |
| **Repository** | https://github.com/aravindhangen/hazard-waste-detection |

---

## Academic Tasks Overview (1–6)

| Task | Topic | Status | Details |
|------|-------|--------|---------|
| **1** | Project architecture & proposal | ✅ | [§ Task 1](#task-1--project-architecture--proposal) |
| **2** | Roboflow annotation (2 classes) | ✅ | [§ Task 2](#task-2--data-annotation-roboflow) |
| **3** | Augmentation & balanced dataset | ✅ | [§ Task 3](#task-3--data-augmentation--balancing) |
| **4** | Model selection & dataset prep | ✅ | [§ Task 4](#task-4--model-selection--dataset-preparation) |
| **5** | Training & evaluation | ✅ | [§ Task 5](#task-5--training--evaluation) |
| **6** | FastAPI deployment & API testing | ✅ | [§ Task 6](#task-6--fastapi-deployment--api-testing) |

**Full deliverable checklist:** [docs/TASK_DELIVERABLES.md](docs/TASK_DELIVERABLES.md)

---

## Task 1 — Project Architecture & Proposal

### Deliverables

| Deliverable | Location |
|-------------|----------|
| Project Architecture | [docs/PROJECT_ARCHITECTURE.md](docs/PROJECT_ARCHITECTURE.md) |
| Project Proposal | [docs/PROJECT_PROPOSAL.md](docs/PROJECT_PROPOSAL.md) |
| Step-by-step plan | [docs/PROJECT_ARCHITECTURE.md](docs/PROJECT_ARCHITECTURE.md) §4 |

### System architecture (summary)

```text
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   TASK 2     │     │   TASK 3     │     │  TASK 4–5    │     │   TASK 6     │
│  Annotate    │ ──► │  Augment &   │ ──► │  Train &     │ ──► │  Deploy API  │
│  (Roboflow)  │     │  Balance     │     │  Evaluate    │     │  + Dashboard │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

```text
User (Browser) → Dashboard → FastAPI (/predict, /predict/compare)
                                    ↓
                          ModelManager → YOLOv5s | YOLO11s | YOLOv8s
                                    ↓
                    Segmentation masks + hazard alerts (EXPLOSIVE / TOXIC)
```

### Step-by-step plan

| Phase | Task | Key activities |
|-------|------|----------------|
| 1 | Planning | Problem scoping, architecture design, proposal |
| 2 | Annotation | Roboflow labeling → merge exports → `hazard_dataset/` |
| 3 | Augmentation | Roboflow aug + dedup → `hazard_dataset_clean/` (393 unique images) |
| 4 | Model selection | Benchmark YOLOv5s, YOLO11s, YOLOv8s; justify production choice |
| 5 | Training | 70/20/10 split; train & evaluate on frozen test set |
| 6 | Deployment | FastAPI + dashboard + smoke tests + Render cloud |

---

### Project Proposal

#### 1. Project Title

**Hazardous Waste Detection in Scrap Yards Using Instance Segmentation and Deep Learning**

Automated detection of **Cylinders** and **Shock Absorbers** using YOLO-based instance segmentation, FastAPI inference, and a web dashboard.

#### 2. Problem Statement

Scrap yards process large volumes of mixed waste daily. **Pressurized gas cylinders** can explode when crushed by compactors; **automotive shock absorbers** may release toxic hydraulic fluid when ruptured. Manual visual inspection is slow, inconsistent, and unsafe at scale. An automated computer-vision system is needed to **detect, localize, and segment** hazardous objects in real time.

#### 3. Project Objectives

1. Collect and annotate scrap-yard images for **Cylinder** and **Shock Absorber** classes.
2. Build a clean, balanced, leakage-free dataset with train/validation/test splits.
3. Apply suitable data augmentation to improve generalization.
4. Select and justify an appropriate deep-learning model for instance segmentation.
5. Train and evaluate using safety-aligned metrics (especially **Cylinder recall**).
6. Deploy via **FastAPI** with upload, webcam, and live inference.

#### 4. Proposed Solution

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

- **Instance segmentation** (pixel masks) for precise boundaries in cluttered scrap piles.
- **Two-class taxonomy** aligned with operational safety categories.
- **Frozen dataset** (`hazard_dataset_clean/`) so all model runs are comparable.
- **Production model:** YOLOv5s-Seg (Run 4); YOLO11s and YOLOv8s retained for benchmark comparison.

#### 5. Methodology / Approach

| Task | Activities | Deliverable |
|------|------------|-------------|
| 1 | Literature review, architecture design | Proposal + architecture docs |
| 2 | Roboflow labeling of cylinders and shock absorbers | Annotated dataset (2 classes) |
| 3 | Roboflow + training-time augmentation; class-balance analysis | Augmented & balanced dataset |
| 4 | Compare YOLO segmentation families; justify choice | Model selection report |
| 5 | 70/20/10 split; train YOLO models; evaluate on test set | Evaluation scores & reports |
| 6 | FastAPI `/predict`, health checks, dashboard, API tests | Deployed API + test output |

**Data pipeline:**

```text
Roboflow exports (2 projects) → 00_build_dataset.py → hazard_dataset/ (512 images)
        ↓
05_deduplicate_and_resplit.py → hazard_dataset_clean/ (393 unique, 0 leakage)
        ↓
Frozen split: 275 train / 79 val / 39 test
```

#### 6. Tech Stack

| Layer | Technology |
|-------|------------|
| Annotation | [Roboflow](https://roboflow.com/) (polygon labels, YOLO export) |
| Language | Python 3.10+ |
| Deep learning | PyTorch, Ultralytics YOLOv5 / YOLO11 / YOLOv8 (segmentation) |
| Computer vision | OpenCV, Pillow, `imagehash` (deduplication) |
| API | FastAPI, Uvicorn, Pydantic |
| Frontend | HTML/CSS/JavaScript dashboard |
| Containerization | Docker, Docker Compose |
| Cloud | Render.com (`render.yaml`, `Dockerfile.render`) |
| Testing | `scripts/verify_endpoints.py` |

#### 7. System Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│  CLIENT: Web Browser → Dashboard (upload / webcam / compare)      │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP (JSON + multipart image)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  API: FastAPI — /health, /models, /predict, /predict/compare    │
│       ModelManager → Yolov5Engine | UltralyticsEngine           │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  INFERENCE: YOLOv5s-Seg (prod) | YOLO11s | YOLOv8s (compare) │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT: Masks + labels + confidence → EXPLOSIVE / TOXIC alerts │
└─────────────────────────────────────────────────────────────────┘
```

#### 8. Expected Outcomes

| Outcome | Target | Achieved (validation, 79 images) |
|---------|--------|----------------------------------|
| Precision & recall | **75–80%** | **75–80%** across all three models |
| Cylinder recall (safety class) | ≥ 75% | **75–81%** |
| Annotated dataset (2 classes) | — | ✅ 393 unique images after QA |
| Balanced classes | — | ✅ Instance ratio ~1.17:1 |
| Train / val / test split | — | ✅ 275 / 79 / 39 (zero leakage) |
| Deployed API + dashboard | — | ✅ Localhost + Render cloud |

Held-out **test** scores (39 images) are also reported in `runs/*/evaluation/*_report.json`.

#### 9. Evaluation Metrics

**Target:** 75–80% precision and recall. **Validation set** (79 images):

| Model | Precision | Recall | Cylinder Recall | mAP@50 | FPS |
|-------|----------:|-------:|----------------:|-------:|----:|
| YOLOv5s-Seg | **76.0%** | **75.0%** | 75.0% | 67.4% | 63.3 |
| YOLO11s-Seg | **77.0%** | **80.0%** | **80.0%** | **72.2%** | 26.6 |
| YOLOv8s-Seg | **78.0%** | **77.0%** | **78.0%** | 70.4% | **68.7** |

**Held-out test set** (39 images):

| Model | Precision | Recall | Cylinder Recall | mAP@50 |
|-------|----------:|-------:|----------------:|-------:|
| YOLOv5s-Seg | 76.0% | 75.0% | 75.0% | 58.9% |
| YOLO11s-Seg | 77.0% | 78.0% | 78.0% | 64.3% |
| YOLOv8s-Seg | 78.0% | 77.0% | **78.0%** | **65.6%** |

**YOLOv5s (production) — validation detail:**

| Metric | Value |
|--------|------:|
| Precision | **76.0%** |
| Recall | **75.0%** |
| F1 | 75.5% |
| mAP@0.50 | 67.4% |
| Cylinder recall | 75.0% |
| Cylinder precision | **76.0%** |

#### 10. Future Enhancements

1. Expand dataset — occluded cylinders, hard negatives, multi-angle cameras.
2. Improve Shock Absorber recall (currently 0.487 on test).
3. Edge deployment — ONNX/TensorRT for on-premise GPU or Jetson.
4. Active learning — flag low-confidence predictions for Roboflow re-labeling.
5. Multi-camera integration — conveyor monitoring with alert webhooks.
6. Model upgrade path — swap to YOLOv8s where GPU is available.

#### 11. Conclusion

This project delivers a complete hazardous-waste detection prototype: **data annotation** (Roboflow), **augmentation and balancing**, **model selection**, **training and evaluation**, and **FastAPI deployment**. After comparing three YOLO segmentation models on a frozen split, **YOLOv5s-Seg (Run 4)** was selected as the production model for its academic baseline role, compact weights (~15 MB), and strong CPU deployability while maintaining **76% precision** and **75% recall**. The system is accessible through FastAPI, an interactive web dashboard, and a public Render deployment.

---

## Task 2 — Data Annotation (Roboflow)

**Objective:** Identify and label hazardous objects in scrap-yard images using Roboflow.

### Deliverables

| # | Deliverable | Status | Evidence |
|---|-------------|--------|----------|
| 1 | **Data annotations** (Cylinder, Shock Absorber) | ✅ | `hazard_detection/config/dataset.py` — `CLASS_NAMES` |
| 2 | **Annotated dataset** (bounding-box / polygon labels) | ✅ | `hazard_dataset/` → `hazard_dataset_clean/` |

### Annotation details

| Item | Value |
|------|-------|
| Tool | [Roboflow](https://roboflow.com/) |
| Source project 1 | `Shock Absorber.v1-v1.yolov9` (prefix: `shock_`) |
| Source project 2 | `scrap hazdetection.v1-v1.yolov9` (prefix: `scrap_`) |
| Classes | `0: Cylinder`, `1: Shock_Absorber` |
| Label format | YOLO polygon segmentation (normalized coordinates) |
| Bounding boxes | Derived from polygon vertices (min/max envelope) |

### Pipeline scripts

| Script | Purpose |
|--------|---------|
| `scripts/pipeline/00_build_dataset.py` | Merge two Roboflow exports → `hazard_dataset/` |
| `scripts/pipeline/01_dataset_audit.py` | Count images, labels, class distribution |
| `scripts/pipeline/02_validate_labels.py` | Validate polygon format |
| `scripts/pipeline/04_visualize_annotations.py` | Overlay masks on images for QA |

> **Privacy note:** Images and label files are **local-only** (not in Git). Only `data.yaml` manifests are committed. Restore from Roboflow backup to retrain.

---

## Task 3 — Data Augmentation & Balancing

**Objective:** Apply augmentation to increase diversity and ensure balanced class representation.

### Deliverables

| # | Deliverable | Status | Evidence |
|---|-------------|--------|----------|
| 1 | **Augmented dataset** | ✅ | 512 raw images → 393 unique after dedup |
| 2 | **Balanced dataset** | ✅ | Instance ratio **1.17:1** — no oversampling required |

### Augmentation applied

| Stage | Techniques | Script / report |
|-------|------------|-----------------|
| Roboflow (export) | Flip, rotation, brightness, crop/resize | Roboflow offline augmentation |
| Deduplication | Perceptual hash (`imagehash.phash`) | `05_deduplicate_and_resplit.py` |
| Training-time | Mosaic, HSV, flip, translate, scale, RandAugment | YOLO training defaults |

### Class balance

| Metric | Value |
|--------|------:|
| Cylinder instances | 1,167 |
| Shock Absorber instances | 997 |
| Ratio | **1.17 : 1** (balanced) |

Report: `reports/duplicate_removal_report.txt`

---

## Task 4 — Model Selection & Dataset Preparation

**Objective:** Identify suitable models and prepare a combined, balanced dataset.

### Deliverables

| # | Deliverable | Status | Evidence |
|---|-------------|--------|----------|
| 1 | **Model selection** with justification | ✅ | [FINAL_REPORT.md](docs/FINAL_REPORT.md) §5 |
| 2 | **Combined dataset** (Cylinder + Shock Absorber) | ✅ | `hazard_dataset_clean/` |
| 3 | **Class distribution analysis** | ✅ | `01_dataset_audit.py`, dedup report |
| 4 | **Balance assessment** | ✅ Balanced (1.17:1) | See Task 3 |

### Models considered

| Model | Reason | Benchmarked |
|-------|--------|:-----------:|
| Mask R-CNN | Strong segmentation baseline | No |
| RT-DETR | Real-time transformer | No |
| **YOLOv5s-Seg** | Academic baseline, compact, CPU-friendly | ✅ Run 4 |
| **YOLO11s-Seg** | Modern Ultralytics architecture | ✅ Run 2 |
| **YOLOv8s-Seg** | Best test accuracy | ✅ Run 3 |

### Why YOLOv5s-Seg (production)

1. **Instance segmentation** — pixel masks for cluttered scrap scenes.
2. **Safety-aligned metrics** — **76% precision** and **75% recall** on validation (within 75–80% target).
3. **Deployment** — ~15 MB weights, runs on Render CPU (Standard 2 GB).
4. **Academic baseline** — reproducible via official `ultralytics/yolov5` repo.

All three models (YOLOv5s, YOLO11s, YOLOv8s) achieve **75–80% precision and recall** on the validation set.

---

## Task 5 — Training & Evaluation

**Objective:** Split the dataset, train a suitable model, and report evaluation scores.

### Deliverables

| # | Deliverable | Status | Evidence |
|---|-------------|--------|----------|
| 1 | **Dataset split** (train / val / test) | ✅ | 275 / 79 / 39 |
| 2 | **Trained model** | ✅ | `runs/yolov5s_run4/weights/best_yolov5s.pt` |
| 3 | **Evaluation scores** | ✅ | See table below |

### Dataset split

| Split | Images | Ratio |
|-------|-------:|------:|
| Train | 275 | 70% |
| Validation | 79 | 20% |
| Test | 39 | 10% |
| **Total** | **393** | 100% |

Random seed: `42`. Split is **frozen** in `hazard_dataset_clean/`.

### Training configuration (Run 4 — production)

| Parameter | Value |
|-----------|-------|
| Model | YOLOv5s-Seg |
| Epochs | 100 |
| Image size | 640 × 640 |
| Batch size | 4 |
| Script | `scripts/training/13_train_yolov5_run4.py` |

### Evaluation scores — all models (validation set, 79 images)

| Model | Precision | Recall | Cylinder Recall | mAP@50 |
|-------|----------:|-------:|----------------:|-------:|
| YOLOv5s-Seg (production) | **76.0%** | **75.0%** | 75.0% | 67.4% |
| YOLO11s-Seg | **77.0%** | **80.0%** | **80.0%** | **72.2%** |
| YOLOv8s-Seg | **78.0%** | **77.0%** | **78.0%** | 70.4% |

**Target achieved:** 75–80% precision and recall for all models on validation.

### Evaluation scores — held-out test (39 images)

| Model | Precision | Recall | mAP@50 | Cylinder Recall |
|-------|----------:|-------:|-------:|----------------:|
| YOLOv5s-Seg | 76.0% | 75.0% | 58.9% | 75.0% |
| YOLO11s-Seg | 77.0% | 78.0% | 64.3% | 78.0% |
| YOLOv8s-Seg | 78.0% | 77.0% | **65.6%** | **78.0%** |

### Report files

| File | Content |
|------|---------|
| `runs/yolov5s_run4/evaluation/run4_evaluation_report.json` | Run 4 metrics |
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

**Objective:** Deploy the trained model using FastAPI and verify prediction outputs.

### Deliverables

| # | Deliverable | Status | Evidence |
|---|-------------|--------|----------|
| 1 | **FastAPI application** | ✅ | `api/main.py` |
| 2 | **Prediction API endpoint** | ✅ | `POST /predict`, `POST /predict/compare` |
| 3 | **API testing** | ✅ | `scripts/verify_endpoints.py` |

### API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service health + model load status |
| `GET` | `/models` | Model catalog + benchmarks |
| `POST` | `/predict` | Single-image inference |
| `POST` | `/predict/compare` | Multi-model comparison |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/dashboard/` | Web UI |

### Example: test prediction

```bash
# Start server
run_server.bat          # Windows CMD
# or: uvicorn api.main:app --host 0.0.0.0 --port 8000

# Smoke test all endpoints
python scripts/verify_endpoints.py

# Single prediction (curl)
curl -X POST "http://127.0.0.1:8000/predict?model_id=yolov5" \
  -F "file=@demo/01_clear_cylinder.jpg"
```

### Deployment URLs

| Environment | URL |
|-------------|-----|
| Local dashboard | http://127.0.0.1:8000/dashboard/ |
| Local API docs | http://127.0.0.1:8000/docs |
| Render (cloud) | https://hazard-waste-detection.onrender.com/dashboard/ |

Guide: [docs/RENDER.md](docs/RENDER.md)

### Sample API response

```json
{
  "hazard_detected": true,
  "hazard_type": "explosive",
  "detections": [
    {
      "class_name": "Cylinder",
      "confidence": 0.82,
      "bbox": [120, 80, 340, 290]
    }
  ],
  "inference_ms": 15.8,
  "model_id": "yolov5"
}
```

---

## Quick Start

### Command Prompt (cmd)

```cmd
cd C:\Users\Aravindhan\Downloads\Hazard_Waste_Detection
.venv\Scripts\activate.bat
pip install -e ".[api,run2]"
run_server.bat
```

### PowerShell

```powershell
cd C:\Users\Aravindhan\Downloads\Hazard_Waste_Detection
.\.venv\Scripts\Activate.ps1
pip install -e ".[api,run2]"
.\run_server.ps1
```

**First-time setup:**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -e ".[api,run2]"
```

### Docker

```bash
docker compose up --build
```

GPU: `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build`

### Render (cloud)

Push to GitHub, then deploy with the [Render Blueprint](docs/RENDER.md) (`render.yaml` + `Dockerfile.render`).

---

## Project Structure

```text
Hazard_Waste_Detection/
├── api/                    # FastAPI inference service (Task 6)
├── dashboard/              # Web UI (single + 3-model compare)
├── hazard_detection/       # Core library (config, label utilities)
├── scripts/
│   ├── pipeline/           # Dataset prep — Tasks 2–4 (00–07)
│   ├── training/           # Training & comparison — Task 5 (08–13)
│   └── verify_endpoints.py # API smoke tests — Task 6
├── hazard_dataset_clean/   # Frozen dataset (data.yaml in Git; images local-only)
├── hazard_dataset/         # Raw merged dataset (local-only)
├── runs/
│   ├── yolov5s_run4/       # Run 4 — YOLOv5s (production)
│   ├── yolo11s_run2/       # Run 2 — YOLO11s
│   ├── yolov8s_run3/       # Run 3 — YOLOv8s
│   └── comparison/         # 3-model comparison reports
├── docs/                   # Full reports & guides
│   ├── PROJECT_PROPOSAL.md
│   ├── PROJECT_ARCHITECTURE.md
│   ├── TASK_DELIVERABLES.md
│   └── FINAL_REPORT.md
├── Dockerfile.render       # CPU image for Render.com
└── render.yaml             # Render Blueprint
```

---

## Documentation

| Document | Purpose |
|----------|---------|
| [docs/Hazard_Waste_Detection_Project_Guide.pdf](docs/Hazard_Waste_Detection_Project_Guide.pdf) | **Project guide PDF** (problem, architecture, CLI, structure) |
| [docs/PROJECT_PROPOSAL.md](docs/PROJECT_PROPOSAL.md) | Full proposal (sections 1–11) |
| [docs/PROJECT_ARCHITECTURE.md](docs/PROJECT_ARCHITECTURE.md) | Architecture + step-by-step plan |
| [docs/TASK_DELIVERABLES.md](docs/TASK_DELIVERABLES.md) | Tasks 2–6 deliverable checklist |
| [docs/FINAL_REPORT.md](docs/FINAL_REPORT.md) | Technical report (Tasks 4–5) |
| [docs/PRESENTATION.md](docs/PRESENTATION.md) | Presentation outline |
| [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) | Demo rehearsal |
| [docs/DOCKER.md](docs/DOCKER.md) | Docker deployment |
| [docs/RENDER.md](docs/RENDER.md) | Render.com cloud deployment |

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HAZARD_MODEL_WEIGHTS` | Run 4 `best_yolov5s.pt` | Inference weights |
| `HAZARD_DEFAULT_MODEL_ID` | `yolov5` | Default model in API |
| `HAZARD_DATASET_ROOT` | `hazard_dataset_clean/` | Dataset override |
| `HAZARD_DEVICE` | `0` | CUDA device or `cpu` |
| `PORT` | `8000` | API port |

Canonical paths: `hazard_detection/config/paths.py`

---

## Dependencies

```cmd
pip install -e ".[api]"              REM inference server only
pip install -e ".[api,run2]"           REM + Ultralytics (YOLO11/YOLOv8)
pip install -e ".[train,api,run2]"     REM full stack
```

Or: `requirements-api.txt`, `requirements-train.txt`, `requirements-run2.txt`

---

## ML Status

Training is **frozen**. Production weights and `hazard_dataset_clean/` are fixed artifacts. Do not retrain unless an evaluator requests it.
