"""Lazy-loaded multi-model inference manager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from api.inference import PredictionResult, SegmentationEngine
from api.ultralytics_engine import UltralyticsEngine
from hazard_detection.config.models import DEFAULT_MODEL_ID, ModelSpec, get_model_catalog, get_model_spec


class InferenceEngine(Protocol):
    weights: object
    device: object

    def predict(
        self,
        image_bgr: np.ndarray,
        conf_thres: float | None = None,
        annotate: bool = True,
    ) -> PredictionResult: ...


@dataclass
class ModelStatus:
    spec: ModelSpec
    loaded: bool
    load_error: str | None = None


class ModelManager:
    def __init__(self) -> None:
        self._engines: dict[str, InferenceEngine] = {}
        self._errors: dict[str, str] = {}

    def list_models(self) -> list[ModelStatus]:
        statuses: list[ModelStatus] = []
        for model_id, spec in get_model_catalog().items():
            statuses.append(
                ModelStatus(
                    spec=spec,
                    loaded=model_id in self._engines,
                    load_error=self._errors.get(model_id),
                )
            )
        return statuses

    def get_spec(self, model_id: str) -> ModelSpec:
        return get_model_spec(model_id)

    def get_engine(self, model_id: str = DEFAULT_MODEL_ID) -> InferenceEngine:
        if model_id in self._engines:
            return self._engines[model_id]

        spec = self.get_spec(model_id)
        if not spec.inference_available:
            raise FileNotFoundError(
                f"Model '{spec.name}' is not available for live inference. "
                f"{spec.benchmark_note or spec.description}"
            )

        try:
            if spec.backend == "yolov9":
                engine: InferenceEngine = SegmentationEngine(weights=spec.weights)
            else:
                engine = UltralyticsEngine(weights=spec.weights)
        except Exception as exc:
            self._errors[model_id] = str(exc)
            raise

        self._engines[model_id] = engine
        self._errors.pop(model_id, None)
        return engine

    def predict(
        self,
        image_bgr: np.ndarray,
        model_id: str = DEFAULT_MODEL_ID,
        conf_thres: float | None = None,
        annotate: bool = True,
    ) -> tuple[ModelSpec, PredictionResult]:
        spec = self.get_spec(model_id)
        engine = self.get_engine(model_id)
        result = engine.predict(image_bgr, conf_thres=conf_thres, annotate=annotate)
        return spec, result

    def compare(
        self,
        image_bgr: np.ndarray,
        model_ids: list[str],
        conf_thres: float | None = None,
        annotate: bool = True,
    ) -> list[tuple[ModelSpec, PredictionResult | None, str | None]]:
        outputs: list[tuple[ModelSpec, PredictionResult | None, str | None]] = []
        for model_id in model_ids:
            spec = self.get_spec(model_id)
            if not spec.inference_available:
                outputs.append((spec, None, spec.benchmark_note or "Live inference unavailable."))
                continue
            try:
                _, result = self.predict(
                    image_bgr,
                    model_id=model_id,
                    conf_thres=conf_thres,
                    annotate=annotate,
                )
                outputs.append((spec, result, None))
            except Exception as exc:
                outputs.append((spec, None, str(exc)))
        return outputs

    @property
    def default_device(self) -> str:
        try:
            engine = self.get_engine(DEFAULT_MODEL_ID)
            return str(engine.device)
        except Exception:
            return "unavailable"
