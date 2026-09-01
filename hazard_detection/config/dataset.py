"""Dataset class definitions, split ratios, and Roboflow source paths."""

from __future__ import annotations

import os
from pathlib import Path

CLASS_NAMES: dict[int, str] = {
    0: "Cylinder",
    1: "Shock_Absorber",
}

SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
RANDOM_STATE = 42

# Raw Roboflow exports used to build hazard_dataset.
# Override with SHOCK_DATASET_PATH / SCRAP_DATASET_PATH for other machines.
SOURCE_DATASETS = [
    {
        "path": Path(
            os.environ.get(
                "SHOCK_DATASET_PATH",
                r"C:\Users\Aravindhan\Downloads\Shock Absorber.v1-v1.yolov9",
            )
        ),
        "prefix": "shock",
    },
    {
        "path": Path(
            os.environ.get(
                "SCRAP_DATASET_PATH",
                r"C:\Users\Aravindhan\Downloads\scrap hazdetection.v1-v1.yolov9",
            )
        ),
        "prefix": "scrap",
    },
]
