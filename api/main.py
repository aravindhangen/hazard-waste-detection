from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from api.inference import SegmentationEngine
from api.model_manager import ModelManager
from api.schemas import (
    BenchmarkMetricsResponse,
    ClassCount,
    CompareResponse,
    HealthResponse,
    ModelCompareItem,
    ModelInfoResponse,
    ModelSummaryResponse,
    ModelsListResponse,
    PredictResponse,
)
from hazard_detection.config import DASHBOARD_DIR
from hazard_detection.config.api_settings import (
    BACKGROUND_WARMUP,
    CONF_THRESHOLD,
    DATA_YAML_PATH,
    DEVICE,
    EAGER_LOAD,
    IMG_SIZE,
    IOU_THRESHOLD,
    MODEL_INFO,
    WEIGHTS_PATH,
)
from hazard_detection.config.models import (
    COMPARISON_REPORT,
    COMPARE_MODEL_ORDER,
    DEFAULT_MODEL_ID,
    LEGACY_COMPARISON_REPORT,
    get_model_catalog,
)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logger = logging.getLogger(__name__)
manager: ModelManager | None = None
_warmup_done = threading.Event()
_warmup_error: str | None = None


def _benchmark_response(spec) -> BenchmarkMetricsResponse:
    bench = spec.benchmark
    return BenchmarkMetricsResponse(
        cylinder_recall=bench.cylinder_recall,
        map50=bench.map50,
        map50_95=bench.map50_95,
        recall=bench.recall,
        precision=bench.precision,
        f1=bench.f1,
        fps=bench.fps,
    )


def _model_summary(status) -> ModelSummaryResponse:
    spec = status.spec
    return ModelSummaryResponse(
        id=spec.id,
        name=spec.name,
        short_name=spec.short_name,
        role=spec.role,
        description=spec.description,
        badge=spec.badge,
        inference_available=spec.inference_available,
        loaded=status.loaded,
        load_error=status.load_error,
        benchmark=_benchmark_response(spec),
        benchmark_note=spec.benchmark_note,
        weights_path=str(spec.weights),
    )


def _build_predict_response(spec, result) -> PredictResponse:
    annotated_base64 = None
    if result.annotated_image is not None:
        annotated_base64 = SegmentationEngine.encode_image_base64(result.annotated_image)

    return PredictResponse(
        model_id=spec.id,
        model_name=spec.name,
        hazard_detected=result.hazard_detected,
        hazard_summary=result.hazard_summary,
        class_counts=[
            ClassCount(class_name=name, count=count)
            for name, count in sorted(result.class_counts.items())
        ],
        detections=result.detections,
        inference_ms=result.inference_ms,
        image_width=result.image_width,
        image_height=result.image_height,
        annotated_image_base64=annotated_base64,
    )


def _recommendation_text() -> str:
    for path in (COMPARISON_REPORT, LEGACY_COMPARISON_REPORT):
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if "best_cylinder_recall" in data:
                best = data["best_cylinder_recall"]
                map50 = best.get("map50", 0.732)
                cyl = best.get("cylinder_recall", 0.734)
                fastest = max(data.get("runs", []), key=lambda r: r.get("fps") or 0, default={})
                fastest_name = fastest.get("model", "YOLOv8s-Seg")
                fastest_fps = fastest.get("fps", 68.7)
                return (
                    f"Production: YOLOv5s-Seg — academic baseline (Run 4). "
                    f"Best mAP@50 in comparison: {map50:.3f} ({best.get('model', 'YOLOv5s-Seg')}). "
                    f"{fastest_name} is fastest ({fastest_fps:.1f} FPS)."
                )
            return data.get("recommendation", "YOLOv5s-Seg is the production model.")
    return (
        "YOLOv5s-Seg is the production model — cloud-friendly academic baseline "
        "compared with YOLO11s and YOLOv8s."
    )


def _background_warmup() -> None:
    global _warmup_error
    try:
        assert manager is not None
        logger.info("Background warmup: loading %s", DEFAULT_MODEL_ID)
        manager.get_engine(DEFAULT_MODEL_ID)
        logger.info("Background warmup complete")
    except Exception as exc:
        _warmup_error = str(exc)
        logger.exception("Background warmup failed")
    finally:
        _warmup_done.set()


def _ensure_inference_ready(model_id: str = DEFAULT_MODEL_ID) -> None:
    if manager is None:
        raise HTTPException(status_code=503, detail="Model manager is not loaded.")

    if manager.is_loaded(model_id):
        return

    if BACKGROUND_WARMUP and not _warmup_done.is_set() and model_id == DEFAULT_MODEL_ID:
        raise HTTPException(
            status_code=503,
            detail="Default model is still loading. Please wait 30–90 seconds and try again.",
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    global manager, _warmup_error
    manager = ModelManager()
    _warmup_error = None
    _warmup_done.clear()
    if EAGER_LOAD:
        manager.get_engine(DEFAULT_MODEL_ID)
        _warmup_done.set()
    elif BACKGROUND_WARMUP:
        threading.Thread(target=_background_warmup, daemon=True, name="model-warmup").start()
    else:
        _warmup_done.set()
    yield
    manager = None


app = FastAPI(
    title="Hazard Waste Detection API",
    description="Multi-model YOLO segmentation inference for cylinders and shock absorbers.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    available = sum(1 for spec in get_model_catalog().values() if spec.inference_available)
    model_loaded = manager is not None and manager.is_loaded(DEFAULT_MODEL_ID)
    warming = bool(manager is not None and BACKGROUND_WARMUP and not _warmup_done.is_set())
    if _warmup_error:
        status = "degraded"
    elif warming:
        status = "warming"
    else:
        status = "ok"
    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        device=manager.default_device if manager else DEVICE,
        models_available=available,
        warming=warming,
        warmup_error=_warmup_error,
    )


@app.get("/models", response_model=ModelsListResponse)
def list_models() -> ModelsListResponse:
    if manager is None:
        raise HTTPException(status_code=503, detail="Model manager is not loaded.")

    models = [_model_summary(status) for status in manager.list_models()]
    models.sort(key=lambda item: COMPARE_MODEL_ORDER.index(item.id) if item.id in COMPARE_MODEL_ORDER else 99)
    return ModelsListResponse(
        default_model_id=DEFAULT_MODEL_ID,
        recommendation=_recommendation_text(),
        models=models,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info(
    model_id: str = Query(default=DEFAULT_MODEL_ID, description="Model identifier."),
) -> ModelInfoResponse:
    if manager is None:
        raise HTTPException(status_code=503, detail="Model manager is not loaded.")

    try:
        engine = manager.get_engine(model_id)
        spec = manager.get_spec(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return ModelInfoResponse(
        weights_path=str(spec.weights),
        data_yaml_path=str(DATA_YAML_PATH),
        device=str(engine.device),
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        image_size=IMG_SIZE,
        model={**MODEL_INFO, "id": spec.id, "name": spec.name, "role": spec.role},
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(..., description="Scrap-yard image (jpg/png/webp)."),
    model_id: str = Query(default=DEFAULT_MODEL_ID, description="Model to run inference with."),
    conf_threshold: float | None = Query(
        default=None,
        ge=0.01,
        le=1.0,
        description="Optional confidence threshold override.",
    ),
    include_annotated_image: bool = Query(
        default=True,
        description="Return base64 annotated image with masks and labels.",
    ),
) -> PredictResponse:
    _ensure_inference_ready(model_id)

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        image_bgr = SegmentationEngine.decode_image(image_bytes)
        spec, result = manager.predict(
            image_bgr,
            model_id=model_id,
            conf_thres=conf_threshold,
            annotate=include_annotated_image,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return _build_predict_response(spec, result)


@app.post("/predict/compare", response_model=CompareResponse)
async def predict_compare(
    file: UploadFile = File(..., description="Scrap-yard image (jpg/png/webp)."),
    model_ids: str = Query(
        default="yolov9,yolo11s,yolov8s",
        description="Comma-separated model ids to compare on the same image.",
    ),
    conf_threshold: float | None = Query(
        default=None,
        ge=0.01,
        le=1.0,
        description="Optional confidence threshold override.",
    ),
    include_annotated_image: bool = Query(
        default=True,
        description="Return base64 annotated images for each model.",
    ),
) -> CompareResponse:
    if manager is None:
        raise HTTPException(status_code=503, detail="Model manager is not loaded.")

    requested_ids = [item.strip() for item in model_ids.split(",") if item.strip()]
    if not requested_ids:
        raise HTTPException(status_code=400, detail="At least one model id is required.")

    for model_id in requested_ids:
        _ensure_inference_ready(model_id)

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        image_bgr = SegmentationEngine.decode_image(image_bytes)
        outputs = manager.compare(
            image_bgr,
            model_ids=requested_ids,
            conf_thres=conf_threshold,
            annotate=include_annotated_image,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {exc}") from exc

    items: list[ModelCompareItem] = []
    image_width = None
    image_height = None

    for spec, result, error in outputs:
        if result is not None:
            image_width = result.image_width
            image_height = result.image_height
            annotated_base64 = None
            if include_annotated_image and result.annotated_image is not None:
                annotated_base64 = SegmentationEngine.encode_image_base64(result.annotated_image)
            items.append(
                ModelCompareItem(
                    model_id=spec.id,
                    model_name=spec.name,
                    role=spec.role,
                    badge=spec.badge,
                    inference_available=spec.inference_available,
                    error=error,
                    hazard_detected=result.hazard_detected,
                    hazard_summary=result.hazard_summary,
                    class_counts=[
                        ClassCount(class_name=name, count=count)
                        for name, count in sorted(result.class_counts.items())
                    ],
                    detections=result.detections,
                    inference_ms=result.inference_ms,
                    image_width=result.image_width,
                    image_height=result.image_height,
                    annotated_image_base64=annotated_base64,
                    benchmark=_benchmark_response(spec),
                )
            )
        else:
            items.append(
                ModelCompareItem(
                    model_id=spec.id,
                    model_name=spec.name,
                    role=spec.role,
                    badge=spec.badge,
                    inference_available=spec.inference_available,
                    error=error,
                    benchmark=_benchmark_response(spec),
                )
            )

    return CompareResponse(models=items, image_width=image_width, image_height=image_height)


@app.get("/")
def root():
    return RedirectResponse(url="/dashboard/")


@app.get("/api")
def api_info() -> dict:
    return {
        "service": "Hazard Waste Detection API",
        "dashboard": "/dashboard/",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "model_info": "/model/info",
            "predict": "POST /predict",
            "predict_compare": "POST /predict/compare",
        },
        "defaults": {
            "model_id": DEFAULT_MODEL_ID,
            "weights": str(WEIGHTS_PATH),
            "data_yaml": str(DATA_YAML_PATH),
            "image_size": IMG_SIZE,
            "conf_threshold": CONF_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
        },
    }


app.mount(
    "/dashboard",
    StaticFiles(directory=str(DASHBOARD_DIR), html=True),
    name="dashboard",
)
