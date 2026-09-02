from __future__ import annotations

import base64
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PredictionResult:
    detections: list[dict]
    class_counts: dict[str, int]
    hazard_detected: bool
    hazard_summary: list[str]
    inference_ms: float
    image_width: int
    image_height: int
    annotated_image: np.ndarray | None = None


def decode_image(image_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image. Supported formats: jpg, png, bmp, webp.")
    return image


def encode_image_base64(image_bgr: np.ndarray, ext: str = ".jpg") -> str:
    success, buffer = cv2.imencode(ext, image_bgr)
    if not success:
        raise ValueError("Failed to encode annotated image.")
    encoded = base64.b64encode(buffer).decode("ascii")
    mime = "image/jpeg" if ext == ".jpg" else "image/png"
    return f"data:{mime};base64,{encoded}"
