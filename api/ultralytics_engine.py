"""Ultralytics YOLO segmentation inference (YOLO11, YOLOv8)."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from api.config import (
    CONF_THRESHOLD,
    DATA_YAML_PATH,
    DEVICE,
    HAZARD_METADATA,
    IMG_SIZE,
    IOU_THRESHOLD,
)
from api.inference import PredictionResult, SegmentationEngine


class UltralyticsEngine:
    def __init__(
        self,
        weights: Path,
        device: str = DEVICE,
        img_size: int = IMG_SIZE,
        conf_thres: float = CONF_THRESHOLD,
        iou_thres: float = IOU_THRESHOLD,
    ) -> None:
        if not weights.exists():
            raise FileNotFoundError(f"Model weights not found: {weights}")

        from ultralytics import YOLO

        self.weights = weights
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.model = YOLO(str(weights))
        self.names = self.model.names

    def predict(
        self,
        image_bgr: np.ndarray,
        conf_thres: float | None = None,
        annotate: bool = True,
    ) -> PredictionResult:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Empty image provided for inference.")

        conf = self.conf_thres if conf_thres is None else conf_thres
        height, width = image_bgr.shape[:2]

        start = time.perf_counter()
        results = self.model.predict(
            source=image_bgr,
            imgsz=self.img_size,
            conf=conf,
            iou=self.iou_thres,
            device=self.device,
            verbose=False,
        )
        inference_ms = (time.perf_counter() - start) * 1000.0

        result = results[0]
        detections: list[dict] = []
        class_counts: dict[str, int] = {}

        if result.boxes is not None and len(result.boxes):
            masks_xy = result.masks.xy if result.masks is not None else [None] * len(result.boxes)
            for box, polygon in zip(result.boxes, masks_xy):
                class_id = int(box.cls[0])
                class_name = self.names[class_id]
                conf_score = float(box.conf[0])
                hazard = HAZARD_METADATA.get(
                    class_name,
                    {
                        "hazard_type": "unknown",
                        "risk": "Unclassified hazardous object.",
                    },
                )
                xyxy = box.xyxy[0].tolist()
                polygon_points = (
                    polygon.reshape(-1, 2).astype(float).tolist()
                    if polygon is not None and len(polygon)
                    else []
                )

                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": round(conf_score, 4),
                        "hazard_type": hazard["hazard_type"],
                        "risk_description": hazard["risk"],
                        "bbox": {
                            "x1": float(xyxy[0]),
                            "y1": float(xyxy[1]),
                            "x2": float(xyxy[2]),
                            "y2": float(xyxy[3]),
                        },
                        "polygon": polygon_points,
                    }
                )
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

        annotated = result.plot() if annotate else None
        hazard_summary = [
            f"{name}: {count} ({HAZARD_METADATA.get(name, {}).get('hazard_type', 'unknown')})"
            for name, count in sorted(class_counts.items())
        ]

        return PredictionResult(
            detections=detections,
            class_counts=class_counts,
            hazard_detected=len(detections) > 0,
            hazard_summary=hazard_summary,
            inference_ms=round(inference_ms, 2),
            image_width=width,
            image_height=height,
            annotated_image=annotated,
        )

    @staticmethod
    def decode_image(image_bytes: bytes) -> np.ndarray:
        return SegmentationEngine.decode_image(image_bytes)

    @staticmethod
    def encode_image_base64(image_bgr: np.ndarray, ext: str = ".jpg") -> str:
        return SegmentationEngine.encode_image_base64(image_bgr, ext)
