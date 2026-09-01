FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    HAZARD_DEVICE=0 \
    PORT=8000 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-train.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

COPY hazard_detection/ ./hazard_detection/
COPY api/ ./api/
COPY dashboard/ ./dashboard/
COPY scripts/render_start.sh ./scripts/render_start.sh
COPY hazard_dataset_clean/data.yaml ./hazard_dataset_clean/data.yaml
COPY yolov9/ ./yolov9/
COPY yolov9/runs/train-seg/hazard_waste_seg/weights/best.pt \
    ./yolov9/runs/train-seg/hazard_waste_seg/weights/best.pt
COPY runs/yolo11s_run2/weights/best_yolo11s.pt \
    ./runs/yolo11s_run2/weights/best_yolo11s.pt
COPY runs/yolov8s_run3/weights/best_yolov8s.pt \
    ./runs/yolov8s_run3/weights/best_yolov8s.pt
COPY runs/comparison/all_runs_comparison.json \
    ./runs/comparison/all_runs_comparison.json

RUN chmod +x ./scripts/render_start.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["./scripts/render_start.sh"]
