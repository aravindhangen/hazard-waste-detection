"""YOLOv5 segmentation inference (Run 4 weights via DetectMultiBackend)."""

from __future__ import annotations

import sys
import time
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
)
from api.inference import PredictionResult
from hazard_detection.config.paths import YOLOV5_DIR

_TORCH_LOAD_PATCHED = False


def _patch_torch_load_for_yolov5() -> None:
    """YOLOv5 checkpoints need full pickle load; PyTorch 2.6+ defaults to weights_only=True."""
    global _TORCH_LOAD_PATCHED
    if _TORCH_LOAD_PATCHED:
        return

    original_load = torch.load

    def load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load_compat  # type: ignore[method-assign]
    _TORCH_LOAD_PATCHED = True


def _ensure_yolov5_on_path() -> Path:
    root = YOLOV5_DIR
    if not root.is_dir():
        raise FileNotFoundError(
            f"YOLOv5 repo not found at {root}. "
            "Clone https://github.com/ultralytics/yolov5 into the project root."
        )
    root_str = str(root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    _patch_torch_load_for_yolov5()
    return root


class Yolov5Engine:
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

        _ensure_yolov5_on_path()

        from models.common import DetectMultiBackend
        from utils.general import check_img_size
        from utils.torch_utils import select_device

        self.weights = weights
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = select_device(device)
        self.model = DetectMultiBackend(
            str(weights),
            device=self.device,
            dnn=False,
            data=str(DATA_YAML_PATH),
            fp16=False,
        )
        self.stride = self.model.stride
        self.names = self.model.names
        self.pt = self.model.pt
        self.img_size = check_img_size(img_size, s=self.stride)

    def predict(
        self,
        image_bgr: np.ndarray,
        conf_thres: float | None = None,
        annotate: bool = True,
    ) -> PredictionResult:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Empty image provided for inference.")

        from ultralytics.utils.ops import masks2segments
        from ultralytics.utils.plotting import Annotator, colors
        from utils.augmentations import letterbox
        from utils.general import non_max_suppression, scale_boxes, scale_segments
        from utils.segment.general import process_mask

        conf = self.conf_thres if conf_thres is None else conf_thres
        height, width = image_bgr.shape[:2]
        im0 = image_bgr.copy()

        im = letterbox(im0, self.img_size, stride=self.stride, auto=self.pt)[0]
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)

        im_tensor = torch.from_numpy(im).to(self.device)
        im_tensor = im_tensor.float() / 255.0
        if im_tensor.ndim == 3:
            im_tensor = im_tensor[None]

        start = time.perf_counter()
        pred, proto = self.model(im_tensor)[:2]
        pred = non_max_suppression(
            pred,
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
        det = pred[0]

        if len(det):
            masks = process_mask(proto[0], det[:, 6:], det[:, :4], im_tensor.shape[2:], upsample=True)
            det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], im0.shape).round()
            segments = [
                scale_segments(im_tensor.shape[2:], segment, im0.shape, normalize=False)
                for segment in masks2segments(masks, strategy="largest")
            ]

            if annotate and annotated is not None:
                annotator = Annotator(annotated, line_width=2, example=str(self.names))
                annotator.masks(masks, colors=[colors(int(cls), True) for cls in det[:, 5]])
                for *xyxy, conf_score, cls in reversed(det[:, :6]):
                    class_id = int(cls)
                    class_name = self._class_name(class_id)
                    label = f"{class_name} {float(conf_score):.2f}"
                    annotator.box_label(xyxy, label, color=colors(class_id, True))
                annotated = annotator.result()

            for index, (*xyxy, conf_score, cls) in enumerate(det[:, :6]):
                class_id = int(cls)
                class_name = self._class_name(class_id)
                hazard = HAZARD_METADATA.get(
                    class_name,
                    {
                        "hazard_type": "unknown",
                        "risk": "Unclassified hazardous object.",
                    },
                )
                polygon = (
                    segments[index].reshape(-1, 2).astype(float).tolist()
                    if index < len(segments) and segments[index] is not None
                    else []
                )
                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": round(float(conf_score), 4),
                        "hazard_type": hazard["hazard_type"],
                        "risk_description": hazard["risk"],
                        "bbox": {
                            "x1": float(xyxy[0]),
                            "y1": float(xyxy[1]),
                            "x2": float(xyxy[2]),
                            "y2": float(xyxy[3]),
                        },
                        "polygon": polygon,
                    }
                )
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

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

    def _class_name(self, class_id: int) -> str:
        if isinstance(self.names, dict):
            return str(self.names[class_id])
        return str(self.names[class_id])
