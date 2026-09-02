"""Runtime settings for the FastAPI inference service."""

from __future__ import annotations

import os
from pathlib import Path

from hazard_detection.config.paths import CLEAN_DATA_YAML, PROJECT_ROOT, RUN4_WEIGHTS


def _resolve_device() -> str:
    """Resolve inference device from env; fall back to CPU when CUDA is unavailable."""
    requested = os.environ.get("HAZARD_DEVICE", "").strip()
    if not requested:
        try:
            import torch

            return "0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    if requested.lower() == "cpu":
        return "cpu"

    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
    except ImportError:
        return "cpu"

    return requested


WEIGHTS_PATH = Path(os.environ.get("HAZARD_MODEL_WEIGHTS", RUN4_WEIGHTS))
DATA_YAML_PATH = Path(os.environ.get("HAZARD_DATA_YAML", CLEAN_DATA_YAML))
DEVICE = _resolve_device()
IMG_SIZE = int(os.environ.get("HAZARD_IMG_SIZE", "640"))
CONF_THRESHOLD = float(os.environ.get("HAZARD_CONF_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.environ.get("HAZARD_IOU_THRESHOLD", "0.45"))

API_HOST = os.environ.get("HAZARD_API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("PORT", os.environ.get("HAZARD_API_PORT", "8000")))
EAGER_LOAD = os.environ.get("HAZARD_EAGER_LOAD", "true").lower() in ("1", "true", "yes")
BACKGROUND_WARMUP = os.environ.get("HAZARD_BACKGROUND_WARMUP", "false").lower() in ("1", "true", "yes")
MAX_LOADED_MODELS = int(os.environ.get("HAZARD_MAX_LOADED_MODELS", "3"))

HAZARD_METADATA = {
    "Cylinder": {
        "hazard_type": "explosive",
        "risk": "Pressurized gas (CNG/LPG/refrigerant); can explode under compression.",
    },
    "Shock_Absorber": {
        "hazard_type": "toxic",
        "risk": "Contains hydraulic oil; must be drained before recycling.",
    },
}

MODEL_INFO = {
    "architecture": "YOLOv5s-Seg",
    "task": "instance_segmentation",
    "classes": ["Cylinder", "Shock_Absorber"],
    "image_size": IMG_SIZE,
    "training_epochs": 100,
}

# Re-export for inference module compatibility.
__all__ = [
    "API_HOST",
    "API_PORT",
    "BACKGROUND_WARMUP",
    "CONF_THRESHOLD",
    "DATA_YAML_PATH",
    "DEVICE",
    "EAGER_LOAD",
    "HAZARD_METADATA",
    "IMG_SIZE",
    "IOU_THRESHOLD",
    "MAX_LOADED_MODELS",
    "MODEL_INFO",
    "PROJECT_ROOT",
    "WEIGHTS_PATH",
]
