# Project Architecture & Step-by-Step Plan

**Task 1 deliverable** — system architecture and implementation roadmap for the Hazardous Waste Detection project.

---

## 1. Project overview

```text
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   TASK 2     │     │   TASK 3     │     │  TASK 4–5    │     │   TASK 6     │
│  Annotate    │ ──► │  Augment &   │ ──► │  Train &     │ ──► │  Deploy API  │
│  (Roboflow)  │     │  Balance     │     │  Evaluate    │     │  + Dashboard │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

---

## 2. System architecture

### 2.1 End-to-end data flow

```text
                    ┌─────────────────────────────────────┐
                    │         DATA ACQUISITION               │
                    │  Scrap-yard photos / screenshots       │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────┐
                    │      ANNOTATION (Roboflow)             │
                    │  Classes: Cylinder, Shock Absorber   │
                    │  Format: YOLO polygon segmentation   │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────┐
                    │      DATASET PIPELINE (Python)        │
                    │  00_build → 05_dedup → 06_validate   │
                    │  Output: hazard_dataset_clean/       │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────┐
                    │         MODEL TRAINING                 │
                    │  YOLOv5s / YOLO11s / YOLOv8s (seg)  │
                    │  Augmentation: mosaic, HSV, flip…    │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────┐
                    │      INFERENCE & DEPLOYMENT            │
                    │  FastAPI → ModelManager → Engines    │
                    │  Dashboard + Render cloud hosting    │
                    └─────────────────────────────────────┘
```

### 2.2 Runtime inference architecture (Task 6)

```text
 User (Browser)
      │
      │  upload image / webcam frame
      ▼
┌─────────────────┐
│   dashboard/    │  app.js — UI, compare mode, threshold slider
│   index.html    │
└────────┬────────┘
         │  POST /predict  or  POST /predict/compare
         ▼
┌─────────────────┐
│    api/main.py  │  FastAPI routes, CORS, static files
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ model_manager.py│  Lazy/eager load, LRU cache, model routing
└────────┬────────┘
         │
    ┌────┴────┬──────────────┐
    ▼         ▼              ▼
┌────────┐ ┌──────────┐ ┌──────────┐
│yolov5_ │ │ultralytics│ │ultralytics│
│engine  │ │ YOLO11s  │ │ YOLOv8s  │
└───┬────┘ └────┬─────┘ └────┬─────┘
    │           │            │
    └───────────┴────────────┘
                │
                ▼
         Masks + boxes + classes
                │
                ▼
         Hazard alert JSON + annotated JPEG
```

### 2.3 Module responsibilities

| Module | Path | Responsibility |
|--------|------|----------------|
| **Config** | `hazard_detection/config/` | Paths, class names, model catalog, API settings |
| **Label utils** | `hazard_detection/data/labels.py` | Polygon validation, cleaning, class counting |
| **Pipeline** | `scripts/pipeline/00–07` | Merge, audit, dedup, visualize, validate |
| **Training** | `scripts/training/08–13` | Train runs, evaluate, compare models |
| **API** | `api/main.py` | HTTP endpoints, request/response schemas |
| **Inference** | `api/inference.py` | Image decode/resize, mask overlay, hazard logic |
| **YOLOv5 engine** | `api/yolov5_engine.py` | Run 4 weights via official YOLOv5 repo |
| **Ultralytics engine** | `api/ultralytics_engine.py` | YOLO11s / YOLOv8s inference |
| **Dashboard** | `dashboard/` | Single + compare UI, benchmark table |
| **Deploy** | `Dockerfile`, `Dockerfile.render`, `render.yaml` | Container + cloud blueprint |

---

## 3. Dataset architecture

```text
hazard_dataset_clean/
├── data.yaml              ← YOLO training manifest (in Git)
├── images/                ← local only (not in Git — privacy)
│   ├── train/   (275)
│   ├── val/     (79)
│   └── test/    (39)
└── labels/                ← local only (not in Git)
    ├── train/
    ├── val/
    └── test/
```

**Class mapping** (`hazard_detection/config/dataset.py`):

| ID | Class | Hazard type |
|----|-------|-------------|
| 0 | Cylinder | Explosive |
| 1 | Shock_Absorber | Toxic |

---

## 4. Step-by-step implementation plan

### Phase 1 — Planning (Task 1) ✅

| Step | Action | Status |
|------|--------|--------|
| 1.1 | Define problem: hazardous objects in scrap yards | Done |
| 1.2 | Choose instance segmentation over plain detection | Done |
| 1.3 | Design API + dashboard architecture | Done |
| 1.4 | Document proposal and architecture | Done — this file + `PROJECT_PROPOSAL.md` |

### Phase 2 — Annotation (Task 2) ✅

| Step | Action | Script / artifact |
|------|--------|-------------------|
| 2.1 | Create Roboflow projects for each hazard type | Shock Absorber.v1, scrap hazdetection.v1 |
| 2.2 | Label objects with polygon / bounding tools | Roboflow UI |
| 2.3 | Export as YOLOv9 segmentation format | Local Roboflow exports |
| 2.4 | Merge exports into unified dataset | `scripts/pipeline/00_build_dataset.py` |
| 2.5 | Audit labels and pairing | `01_dataset_audit.py`, `02_validate_labels.py` |
| 2.6 | Visualize annotations | `04_visualize_annotations.py` |

### Phase 3 — Augmentation & balancing (Task 3) ✅

| Step | Action | Details |
|------|--------|---------|
| 3.1 | Roboflow offline augmentation | Flip, rotation, brightness — contributes to 512 raw images |
| 3.2 | Deduplicate augmented copies | `05_deduplicate_and_resplit.py` (perceptual hash) → 393 unique |
| 3.3 | Analyze class distribution | Instance ratio **1,167 : 997** (~1.17:1) — balanced |
| 3.4 | Training-time augmentation | Mosaic, horizontal flip, HSV jitter, translate, scale (YOLO defaults) |
| 3.5 | Stratified resplit | 70% train / 20% val / 10% test, `RANDOM_STATE=42` |

### Phase 4 — Model selection (Task 4) ✅

| Step | Action | Outcome |
|------|--------|---------|
| 4.1 | Shortlist architectures | Mask R-CNN, RT-DETR, YOLO-Seg family |
| 4.2 | Benchmark three YOLO models on frozen test set | YOLOv5s, YOLO11s, YOLOv8s |
| 4.3 | Prioritize Cylinder recall (safety) | YOLOv8s best (0.720); YOLOv5s competitive (0.710) |
| 4.4 | Select production model | **YOLOv5s-Seg** — baseline, ~15 MB, CPU-friendly |
| 4.5 | Document rationale | `PROJECT_PROPOSAL.md` §5, `FINAL_REPORT.md` §5 |

### Phase 5 — Training & evaluation (Task 5) ✅

| Step | Action | Script |
|------|--------|--------|
| 5.1 | Split dataset 275/79/39 | Frozen in `hazard_dataset_clean/` |
| 5.2 | Train YOLOv5s Run 4 (100 epochs, 640px) | `13_train_yolov5_run4.py` |
| 5.3 | Train YOLO11s Run 2, YOLOv8s Run 3 | `09_train_yolo11_run2.py`, `11_train_yolov8_run3.py` |
| 5.4 | Evaluate on val + test | `runs/*/evaluation/*_report.json` |
| 5.5 | Compare all runs | `12_compare_all_runs.py` |

### Phase 6 — Deployment (Task 6) ✅

| Step | Action | Artifact |
|------|--------|----------|
| 6.1 | Implement FastAPI app | `api/main.py` |
| 6.2 | `POST /predict` endpoint | Single-model inference |
| 6.3 | `POST /predict/compare` | Multi-model side-by-side |
| 6.4 | Health + model catalog | `GET /health`, `GET /models` |
| 6.5 | Web dashboard | `dashboard/` |
| 6.6 | API smoke tests | `scripts/verify_endpoints.py` |
| 6.7 | Docker + Render deploy | `Dockerfile.render`, `render.yaml` |

---

## 5. Deployment topology

```text
┌─────────────────────────────────────────────────────────┐
│                    LOCAL DEVELOPMENT                       │
│  run_server.bat / run_server.ps1  →  localhost:8000      │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    DOCKER (optional)                     │
│  docker compose up  →  GPU or CPU container              │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    RENDER.COM (production demo)          │
│  Dockerfile.render  →  Standard 2GB  →  CPU inference   │
│  https://hazard-waste-detection.onrender.com             │
└─────────────────────────────────────────────────────────┘
```

---

## 6. Configuration reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `HAZARD_DEFAULT_MODEL_ID` | `yolov5` | Production model |
| `HAZARD_MODEL_WEIGHTS` | Run 4 `best_yolov5s.pt` | Weight path override |
| `HAZARD_DATASET_ROOT` | `hazard_dataset_clean/` | Training data root |
| `HAZARD_DEVICE` | `0` / `cpu` | Inference device |
| `HAZARD_IMG_SIZE` | `640` (416 on Render) | Input resolution |
| `PORT` | `8000` | API port |

---

## 7. Related documents

- [PROJECT_PROPOSAL.md](PROJECT_PROPOSAL.md) — formal proposal (Tasks 1–6 summary)
- [TASK_DELIVERABLES.md](TASK_DELIVERABLES.md) — per-task deliverable checklist
- [FINAL_REPORT.md](FINAL_REPORT.md) — evaluation results and limitations
- [RENDER.md](RENDER.md) — cloud deployment guide
