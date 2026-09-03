FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    HAZARD_DEVICE=0 \
    HAZARD_DEFAULT_MODEL_ID=yolov5 \
    PORT=8000 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-train.txt requirements-api.txt requirements-run2.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt -r requirements-run2.txt

RUN git clone --depth 1 --branch v7.0 https://github.com/ultralytics/yolov5.git /app/yolov5

COPY hazard_detection/ ./hazard_detection/
COPY api/ ./api/
COPY dashboard/ ./dashboard/
COPY scripts/render_start.sh ./scripts/render_start.sh
COPY hazard_dataset_clean/data.yaml ./hazard_dataset_clean/data.yaml
COPY runs/yolov5s_run4/weights/best_yolov5s.pt \
    ./runs/yolov5s_run4/weights/best_yolov5s.pt
COPY runs/yolov5s_run4/evaluation/run4_evaluation_report.json \
    ./runs/yolov5s_run4/evaluation/run4_evaluation_report.json
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
