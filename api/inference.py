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


def decode_image(image_bytes: bytes, max_side: int | None = None) -> np.ndarray:
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image. Supported formats: jpg, png, bmp, webp.")
    if max_side and max_side > 0:
        height, width = image.shape[:2]
        longest = max(height, width)
        if longest > max_side:
            scale = max_side / float(longest)
            image = cv2.resize(
                image,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
    return image


def encode_image_base64(image_bgr: np.ndarray, ext: str = ".jpg", quality: int = 82) -> str:
    params = []
    if ext == ".jpg":
        params = [int(cv2.IMWRITE_JPEG_QUALITY), max(40, min(quality, 95))]
    success, buffer = cv2.imencode(ext, image_bgr, params)
    if not success:
        raise ValueError("Failed to encode annotated image.")
    encoded = base64.b64encode(buffer).decode("ascii")
    mime = "image/jpeg" if ext == ".jpg" else "image/png"
    return f"data:{mime};base64,{encoded}"
