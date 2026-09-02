from __future__ import annotations

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    hazard_type: str
    risk_description: str
    bbox: BoundingBox
    polygon: list[list[float]] = Field(
        description="Segmentation polygon as [[x, y], ...] in original image pixel coordinates."
    )


class ClassCount(BaseModel):
    class_name: str
    count: int


class PredictResponse(BaseModel):
    model_id: str = "yolov9"
    model_name: str = "YOLOv9 GELAN-C-SEG"
    hazard_detected: bool
    hazard_summary: list[str]
    class_counts: list[ClassCount]
    detections: list[Detection]
    inference_ms: float
    image_width: int
    image_height: int
    annotated_image_base64: str | None = None


class BenchmarkMetricsResponse(BaseModel):
    cylinder_recall: float | None = None
    map50: float | None = None
    map50_95: float | None = None
    recall: float | None = None
    precision: float | None = None
    f1: float | None = None
    fps: float | None = None


class ModelSummaryResponse(BaseModel):
    id: str
    name: str
    short_name: str
    role: str
    description: str
    badge: str | None = None
    inference_available: bool
    loaded: bool
    load_error: str | None = None
    benchmark: BenchmarkMetricsResponse
    benchmark_note: str | None = None
    weights_path: str


class ModelsListResponse(BaseModel):
    default_model_id: str
    recommendation: str
    models: list[ModelSummaryResponse]


class ModelCompareItem(BaseModel):
    model_id: str
    model_name: str
    role: str
    badge: str | None = None
    inference_available: bool
    error: str | None = None
    hazard_detected: bool | None = None
    hazard_summary: list[str] = Field(default_factory=list)
    class_counts: list[ClassCount] = Field(default_factory=list)
    detections: list[Detection] = Field(default_factory=list)
    inference_ms: float | None = None
    image_width: int | None = None
    image_height: int | None = None
    annotated_image_base64: str | None = None
    benchmark: BenchmarkMetricsResponse


class CompareResponse(BaseModel):
    models: list[ModelCompareItem]
    image_width: int | None = None
    image_height: int | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    models_available: int = 0
    warming: bool = False
    warmup_error: str | None = None


class ModelInfoResponse(BaseModel):
    weights_path: str
    data_yaml_path: str
    device: str
    conf_threshold: float
    iou_threshold: float
    image_size: int
    model: dict
