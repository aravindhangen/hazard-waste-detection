"""Lazy-loaded multi-model inference manager."""

from __future__ import annotations

import gc
import threading
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from api.inference import PredictionResult
from api.ultralytics_engine import UltralyticsEngine
from api.yolov5_engine import Yolov5Engine
from hazard_detection.config.api_settings import DEVICE, MAX_LOADED_MODELS
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
        self._load_lock = threading.Lock()

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

    def is_loaded(self, model_id: str) -> bool:
        return model_id in self._engines

    def _release_engine(self, model_id: str) -> None:
        engine = self._engines.pop(model_id, None)
        if engine is None:
            return
        if hasattr(engine, "model"):
            engine.model = None
        del engine
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _evict_if_needed(self, model_id: str) -> None:
        if MAX_LOADED_MODELS <= 0:
            return
        while len(self._engines) >= MAX_LOADED_MODELS:
            evicted = False
            for cached_id in list(self._engines):
                if cached_id != model_id:
                    self._release_engine(cached_id)
                    evicted = True
                    break
            if not evicted:
                break

    def get_engine(self, model_id: str = DEFAULT_MODEL_ID) -> InferenceEngine:
        if model_id in self._engines:
            return self._engines[model_id]

        with self._load_lock:
            if model_id in self._engines:
                return self._engines[model_id]

            spec = self.get_spec(model_id)
            if not spec.inference_available:
                raise FileNotFoundError(
                    f"Model '{spec.name}' is not available for live inference. "
                    f"{spec.benchmark_note or spec.description}"
                )

            self._evict_if_needed(model_id)

            try:
                if spec.backend == "yolov5":
                    engine: InferenceEngine = Yolov5Engine(weights=spec.weights)
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
        if DEFAULT_MODEL_ID in self._engines:
            return str(self._engines[DEFAULT_MODEL_ID].device)
        return DEVICE
