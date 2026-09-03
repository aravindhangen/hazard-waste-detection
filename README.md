# Hazard Waste Detection

Instance segmentation for **Cylinders** and **Shock Absorbers** in scrap-yard imagery.

**Production model:** YOLOv5s-Seg (Run 4)  
`runs/yolov5s_run4/weights/best_yolov5s.pt`

**Comparison models:** YOLO11s (Run 2), YOLOv8s (Run 3)

## Quick start

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

Or use the batch launcher from PowerShell (note the `.\` prefix):

```powershell
.\run_server.bat
```

**First-time setup** (if `.venv` does not exist):

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -e ".[api,run2]"
```

### URLs

| Service | URL |
|---------|-----|
| Dashboard | http://127.0.0.1:8000/dashboard/ |
| API docs | http://127.0.0.1:8000/docs |
| Health | http://127.0.0.1:8000/health |

Stop the server with **Ctrl+C**.

### Docker

```bash
docker compose up --build
```

GPU: `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build`

### Render (cloud)

Push to GitHub, then deploy with the [Render Blueprint](docs/RENDER.md) (`render.yaml` + `Dockerfile.render`).

- **Plan:** Standard (2 GB RAM) recommended
- **Dashboard:** `https://<your-app>.onrender.com/dashboard/`

See [docs/RENDER.md](docs/RENDER.md) for step-by-step instructions.

## Project structure

```text
Hazard_Waste_Detection/
├── api/                    # FastAPI inference service
├── dashboard/              # Web UI (single + 3-model compare)
├── hazard_detection/       # Core library
│   ├── config/             # paths, dataset, models, API settings
│   └── data/               # Label utilities
├── scripts/
│   ├── pipeline/           # Dataset prep (00–07)
│   ├── training/           # Training & comparison (08–13)
│   ├── verify_endpoints.py # API smoke tests
│   └── _common.py          # Shared script bootstrap
├── hazard_dataset_clean/   # Frozen dataset (393 images)
├── hazard_dataset/         # Raw merged dataset (512 images)
├── runs/
│   ├── yolov5s_run4/       # Run 4 — YOLOv5s (production)
│   ├── yolo11s_run2/       # Run 2 — YOLO11s
│   ├── yolov8s_run3/       # Run 3 — YOLOv8s
│   └── comparison/         # 3-model comparison reports
├── evaluation_reports/     # Legacy Run 1 evaluation (archived)
├── pretrained/             # Ultralytics base checkpoints
├── reports/                # Pipeline logs (dedup report)
├── notebooks/              # Jupyter notebooks
├── demo/                   # Presentation demo images
├── docs/                   # Reports, presentation, Docker guide
├── run_server.bat          # Windows CMD launcher
├── run_server.ps1          # Windows PowerShell launcher
├── Dockerfile              # Local / GPU-capable image
├── Dockerfile.render       # CPU image for Render.com
├── render.yaml             # Render Blueprint
├── docker-compose.yml
```

## Documentation

| Document | Purpose |
|----------|---------|
| [docs/FINAL_REPORT.md](docs/FINAL_REPORT.md) | Final project report |
| [docs/PRESENTATION.md](docs/PRESENTATION.md) | Presentation outline |
| [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) | Demo rehearsal |
| [docs/DOCKER.md](docs/DOCKER.md) | Docker deployment |
| [docs/RENDER.md](docs/RENDER.md) | Render.com cloud deployment |

## Scripts

Run from the project root with `PYTHONPATH` set (launchers do this automatically).

| Script | Purpose |
|--------|---------|
| `python scripts/pipeline/00_build_dataset.py` | Merge Roboflow exports |
| `python scripts/pipeline/05_deduplicate_and_resplit.py` | Dedup → `hazard_dataset_clean/` |
| `python scripts/training/13_train_yolov5_run4.py` | Run 4 — YOLOv5s (production) |
| `python scripts/training/08_train_yolov9.py` | Legacy Run 1 — YOLOv9 (archived) |
| `python scripts/training/09_train_yolo11_run2.py` | Run 2 — YOLO11s |
| `python scripts/training/11_train_yolov8_run3.py` | Run 3 — YOLOv8s |
| `python scripts/training/12_compare_all_runs.py` | 3-model comparison |
| `python scripts/verify_endpoints.py` | API smoke tests |

## Models (frozen test set)

| Model | mAP@50 | Cylinder Recall | FPS | Role |
|-------|--------|-----------------|-----|------|
| YOLOv5s-Seg | 0.589 | 0.710 | 63.3 | **Production** |
| YOLO11s-Seg | 0.643 | 0.714 | 26.6 | Tested |
| YOLOv8s-Seg | 0.656 | 0.720 | 68.7 | Tested (best accuracy) |

## Dashboard

- **Single Model** — one model on upload/webcam
- **Compare Models** — side-by-side outputs for all 3 models + PNG export
- **Benchmark table** — held-out test metrics

## Configuration

Canonical paths: `hazard_detection/config/paths.py`

| Variable | Default | Description |
|----------|---------|-------------|
| `HAZARD_MODEL_WEIGHTS` | Run 4 `best_yolov5s.pt` | Inference weights |
| `HAZARD_DEFAULT_MODEL_ID` | `yolov5` | Default model in API |
| `HAZARD_DATASET_ROOT` | `hazard_dataset_clean/` | Dataset override |
| `HAZARD_DEVICE` | `0` | CUDA device or `cpu` |
| `PORT` | `8000` | API port |

## Dependencies

```cmd
pip install -e ".[api]"              REM inference server only
pip install -e ".[api,run2]"           REM + Ultralytics (YOLO11/YOLOv8)
pip install -e ".[train,api,run2]"     REM full stack
```

Or: `requirements-api.txt`, `requirements-train.txt`, `requirements-run2.txt`

## ML status

Training is **frozen**. Production weights and `hazard_dataset_clean/` are fixed artifacts. Do not retrain unless an evaluator requests it.
