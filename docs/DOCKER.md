# Docker Deployment Guide

## Quick start (CPU — recommended for evaluators)

```bash
docker compose up --build
```

Open: http://localhost:8000/dashboard/

Stop:

```bash
docker compose down
```

CPU inference is slower than the RTX 4050 training setup but sufficient for academic demonstration.

---

## GPU deployment (optional)

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

---

## What is included in the image

| Component | Included |
|-----------|----------|
| YOLOv9 production weights (`best.pt`) | Yes |
| YOLO11s comparison weights | Yes (if present at build time) |
| FastAPI + dashboard | Yes |
| Full dataset images | No (only `data.yaml` for class names) |
| Training scripts | No |

---

## Environment variables

| Variable | Default (CPU compose) | Description |
|----------|----------------------|-------------|
| `HAZARD_DEVICE` | `cpu` | `cpu` or CUDA device index `0` |
| `PORT` | `8000` | API port |
| `HAZARD_MODEL_WEIGHTS` | Run 1 `best.pt` | Override production weights |
| `KMP_DUPLICATE_LIB_OK` | `TRUE` | OpenMP workaround |

---

## Health check

```bash
curl http://localhost:8000/health
curl http://localhost:8000/models
```

---

## Build notes

- Base image: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` (runs on CPU when `HAZARD_DEVICE=cpu`)
- First startup loads YOLOv9 (~30–90 s depending on hardware)
- Image size ~3–4 GB with weights

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Port 8000 in use | Change `ports: "8080:8000"` in compose file |
| GPU not detected | Use CPU compose or install NVIDIA Container Toolkit |
| Model compare missing YOLO11s | Rebuild after ensuring `runs/yolo11s_run2/weights/best_yolo11s.pt` exists |
| Slow inference on CPU | Expected; mention training was on RTX 4050, deployment can be CPU |

---

## Production vs experimental in Docker

- **Default `/predict`** always uses YOLOv9 unless `model_id` is specified
- **Compare mode** in dashboard uses `/predict/compare` for YOLOv9 + YOLO11s demo only
- YOLOv8s remains unavailable until Run 3 weights are added
