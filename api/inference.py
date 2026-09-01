from __future__ import annotations

import base64
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from api.config import (
    CONF_THRESHOLD,
    DATA_YAML_PATH,
    DEVICE,
    HAZARD_METADATA,
    IMG_SIZE,
    IOU_THRESHOLD,
    WEIGHTS_PATH,
    YOLOV9_DIR,
)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

if str(YOLOV9_DIR) not in sys.path:
    sys.path.insert(0, str(YOLOV9_DIR))

from models.common import DetectMultiBackend  # noqa: E402
from utils.augmentations import letterbox  # noqa: E402
from utils.general import non_max_suppression  # noqa: E402
from utils.plots import Annotator, colors  # noqa: E402
from utils.segment.general import masks2segments, process_mask  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402


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


class SegmentationEngine:
    def __init__(
        self,
        weights: Path = WEIGHTS_PATH,
        data_yaml: Path = DATA_YAML_PATH,
        device: str = DEVICE,
        img_size: int = IMG_SIZE,
        conf_thres: float = CONF_THRESHOLD,
        iou_thres: float = IOU_THRESHOLD,
    ) -> None:
        if not weights.exists():
            raise FileNotFoundError(f"Model weights not found: {weights}")
        if not data_yaml.exists():
            raise FileNotFoundError(f"Dataset config not found: {data_yaml}")

        self.weights = weights
        self.data_yaml = data_yaml
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = select_device(device)

        self.model = DetectMultiBackend(
            str(weights),
            device=self.device,
            dnn=False,
            data=str(data_yaml),
            fp16=False,
        )
        self.stride = self.model.stride
        self.names = self.model.names
        self.model.warmup(imgsz=(1, 3, img_size, img_size))

    def _preprocess(self, image_bgr: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        im = letterbox(image_bgr, self.img_size, stride=self.stride, auto=self.model.pt)[0]
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)
        tensor = torch.from_numpy(im).to(self.model.device)
        tensor = tensor.float() / 255.0
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        return tensor, image_bgr

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
        tensor, im0 = self._preprocess(image_bgr)

        start = time.perf_counter()
        preds, train_out = self.model(tensor, augment=False)
        protos = train_out[-1]
        pred = non_max_suppression(
            preds,
            conf,
            self.iou_thres,
            classes=None,
            agnostic=False,
            max_det=1000,
            nm=32,
        )
        inference_ms = (time.perf_counter() - start) * 1000.0

        detections: list[dict] = []
        class_counts: dict[str, int] = {}
        annotated = im0.copy() if annotate else None
        annotator = Annotator(annotated, line_width=2, example=str(self.names)) if annotate else None

        det = pred[0]
        proto = protos[0]
        if det is not None and len(det):
            masks = process_mask(proto, det[:, 6:], det[:, :4], tensor.shape[2:], upsample=True)
            det[:, :4] = self._scale_boxes(tensor.shape[2:], det[:, :4], im0.shape).round()
            segments = masks2segments(masks)

            if annotate and annotator is not None:
                annotator.masks(
                    masks,
                    colors=[colors(int(cls), True) for cls in det[:, 5]],
                    im_gpu=None,
                )

            for (*xyxy, conf_score, cls_id), segment in zip(det[:, :6], segments):
                class_id = int(cls_id)
                class_name = self.names[class_id]
                hazard = HAZARD_METADATA.get(
                    class_name,
                    {
                        "hazard_type": "unknown",
                        "risk": "Unclassified hazardous object.",
                    },
                )
                polygon = segment.reshape(-1, 2).astype(float).tolist()
                bbox = {
                    "x1": float(xyxy[0]),
                    "y1": float(xyxy[1]),
                    "x2": float(xyxy[2]),
                    "y2": float(xyxy[3]),
                }

                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": round(float(conf_score), 4),
                        "hazard_type": hazard["hazard_type"],
                        "risk_description": hazard["risk"],
                        "bbox": bbox,
                        "polygon": polygon,
                    }
                )
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

                if annotate and annotator is not None:
                    label = f"{class_name} {conf_score:.2f}"
                    annotator.box_label(xyxy, label, color=colors(class_id, True))

        if annotate and annotator is not None:
            annotated = annotator.result()

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
    def _scale_boxes(img_shape, boxes, im0_shape):
        from utils.general import scale_boxes

        return scale_boxes(img_shape, boxes, im0_shape)

    @staticmethod
    def decode_image(image_bytes: bytes) -> np.ndarray:
        array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image. Supported formats: jpg, png, bmp, webp.")
        return image

    @staticmethod
    def encode_image_base64(image_bgr: np.ndarray, ext: str = ".jpg") -> str:
        success, buffer = cv2.imencode(ext, image_bgr)
        if not success:
            raise ValueError("Failed to encode annotated image.")
        encoded = base64.b64encode(buffer).decode("ascii")
        mime = "image/jpeg" if ext == ".jpg" else "image/png"
        return f"data:{mime};base64,{encoded}"
