# Deploy on Render

Host the FastAPI app + dashboard on [Render](https://render.com) using the CPU Docker image (`Dockerfile.render`).

## Requirements

| Item | Notes |
|------|--------|
| **GitHub repo** | Push this project (including model weights) to GitHub |
| **Render plan** | **Standard (2 GB)** — default model `yolov5` on cloud CPU |
| **Weights in repo** | These must exist at build time (see below) |

### Required weight files (committed to Git)

```
runs/yolov5s_run4/weights/best_yolov5s.pt
runs/yolo11s_run2/weights/best_yolo11s.pt
runs/yolov8s_run3/weights/best_yolov8s.pt
runs/comparison/all_runs_comparison.json
```

Optional (for dashboard benchmark details):

```
runs/yolov5s_run4/evaluation/run4_evaluation_report.json
```

```bash
git lfs install
git lfs track "runs/yolov5s_run4/weights/best_yolov5s.pt"
git add .gitattributes
git add runs/yolov5s_run4/weights/best_yolov5s.pt
```

Or host weights on cloud storage and download them in a custom build step.

> **Standard (2 GB)** runs **YOLOv5s** by default on Render (`HAZARD_DEFAULT_MODEL_ID=yolov5` in `render.yaml`).

---

## Option A — Blueprint (recommended)

1. Push the project to GitHub.
2. Go to [Render Dashboard](https://dashboard.render.com/) → **New** → **Blueprint**.
3. Connect the repository and select `render.yaml`.
4. Click **Apply** — Render builds `Dockerfile.render` and deploys the web service.
5. Open `https://<your-service>.onrender.com/dashboard/`

---

## Option B — Manual web service

1. **New** → **Web Service** → connect GitHub repo.
2. Settings:
   - **Language:** Docker
   - **Dockerfile path:** `Dockerfile.render`
   - **Plan:** Standard (2 GB)
   - **Health check path:** `/health`
3. **Environment variables:**

   | Key | Value |
   |-----|--------|
   | `HAZARD_DEVICE` | `cpu` |
   | `HAZARD_DEFAULT_MODEL_ID` | `yolov5` |
   | `KMP_DUPLICATE_LIB_OK` | `TRUE` |
   | `PYTHONPATH` | `/app` |

4. **Create Web Service** and wait for the build (first deploy may take 10–20 minutes).

---

## URLs after deploy

| Endpoint | Path |
|----------|------|
| Dashboard | `https://<app>.onrender.com/dashboard/` |
| API docs | `https://<app>.onrender.com/docs` |
| Health | `https://<app>.onrender.com/health` |

---

## Notes

- **CPU inference** is slower than local GPU (~30–120 s cold start, then seconds per image).
- Render sets `PORT` automatically; `scripts/render_start.sh` binds uvicorn to it.
- **No GPU** on Render standard web services — `HAZARD_DEVICE=cpu` is required.
- Free tier spins down after inactivity; first request after sleep has a long cold start.
- The Docker image clones `ultralytics/yolov5` at build time for Run 4 inference.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Build fails: weight file not found | Ensure `best_yolov5s.pt` and comparison weights exist in the repo |
| OOM / crash on startup | Upgrade to Standard (2 GB) or higher; keep `HAZARD_MAX_LOADED_MODELS=1` |
| Health check fails | Allow 3+ minutes on first boot (model load); increase start period in Render settings |
| 502 after idle | Free tier waking up — retry after 30–60 s |
| Compare models missing | Rebuild after adding Run 2/3 weights to the repo |
| Wrong default model | Set `HAZARD_DEFAULT_MODEL_ID=yolov5` in Render env vars |

---

## Local test (same image as Render)

```bash
docker build -f Dockerfile.render -t hazard-render .
docker run --rm -p 8000:8000 -e PORT=8000 hazard-render
```

Open http://localhost:8000/dashboard/
