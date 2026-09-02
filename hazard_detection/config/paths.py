"""Canonical filesystem paths for datasets, models, and reports."""

from __future__ import annotations

import os
from pathlib import Path

# hazard_detection/config/paths.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Top-level project directories
# ---------------------------------------------------------------------------
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PRETRAINED_DIR = PROJECT_ROOT / "pretrained"
REPORTS_DIR = PROJECT_ROOT / "reports"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
VISUAL_QA_DIR = PROJECT_ROOT / "visual_qa"
DEMO_DIR = PROJECT_ROOT / "demo"

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
DATASET_ROOT = PROJECT_ROOT / "hazard_dataset"
_dataset_override = os.environ.get("HAZARD_DATASET_ROOT")
CLEAN_DATASET_ROOT = (
    Path(_dataset_override) if _dataset_override else PROJECT_ROOT / "hazard_dataset_clean"
)

IMAGE_DIR = DATASET_ROOT / "images"
LABEL_DIR = DATASET_ROOT / "labels"
CLEAN_IMAGE_DIR = CLEAN_DATASET_ROOT / "images"
CLEAN_LABEL_DIR = CLEAN_DATASET_ROOT / "labels"
CLEAN_DATA_YAML = CLEAN_DATASET_ROOT / "data.yaml"

# ---------------------------------------------------------------------------
# YOLOv9 (Run 1 — production)
# ---------------------------------------------------------------------------
YOLOV9_DIR = PROJECT_ROOT / "yolov9"
RUN1_WEIGHTS = (
    YOLOV9_DIR / "runs" / "train-seg" / "hazard_waste_seg" / "weights" / "best.pt"
)
RUN1_TRAIN_DIR = YOLOV9_DIR / "runs" / "train-seg"
RUN1_VAL_DIR = YOLOV9_DIR / "runs" / "val-seg"
EVALUATION_REPORTS_DIR = PROJECT_ROOT / "evaluation_reports"
RUN1_REPORT_JSON = EVALUATION_REPORTS_DIR / "final_evaluation_report.json"
RUN1_REPORT_TXT = EVALUATION_REPORTS_DIR / "final_evaluation_report.txt"

# ---------------------------------------------------------------------------
# YOLOv5 (Run 4 — production / academic baseline)
# ---------------------------------------------------------------------------
RUN4_DIR = PROJECT_ROOT / "runs" / "yolov5s_run4"
RUN4_WEIGHTS = RUN4_DIR / "weights" / "best_yolov5s.pt"
RUN4_REPORT_JSON = RUN4_DIR / "evaluation" / "run4_evaluation_report.json"

# ---------------------------------------------------------------------------
# YOLO11 (Run 2 — experimental comparison)
# ---------------------------------------------------------------------------
RUN2_DIR = PROJECT_ROOT / "runs" / "yolo11s_run2"
RUN2_WEIGHTS = RUN2_DIR / "weights" / "best_yolo11s.pt"
RUN2_REPORT_JSON = RUN2_DIR / "evaluation" / "run2_evaluation_report.json"
COMPARISON_DIR = PROJECT_ROOT / "runs" / "comparison"

# ---------------------------------------------------------------------------
# YOLOv8 (Run 3 — experimental comparison)
# ---------------------------------------------------------------------------
RUN3_DIR = PROJECT_ROOT / "runs" / "yolov8s_run3"
RUN3_WEIGHTS = RUN3_DIR / "weights" / "best_yolov8s.pt"
RUN3_REPORT_JSON = RUN3_DIR / "evaluation" / "run3_evaluation_report.json"

# ---------------------------------------------------------------------------
# Pretrained checkpoints (Ultralytics base weights)
# ---------------------------------------------------------------------------
def _pretrained_path(filename: str) -> Path:
    """Resolve pretrained checkpoint path (new layout with legacy root fallback)."""
    primary = PRETRAINED_DIR / filename
    legacy = PROJECT_ROOT / filename
    if primary.exists():
        return primary
    if legacy.exists():
        return legacy
    return primary


RUN2_PRETRAINED = _pretrained_path("yolo11s-seg.pt")
RUN3_PRETRAINED = _pretrained_path("yolov8s-seg.pt")
RUN4_PRETRAINED = _pretrained_path("yolov5s-seg.pt")

# ---------------------------------------------------------------------------
# Pipeline / QA reports
# ---------------------------------------------------------------------------
DEDUP_REPORT = REPORTS_DIR / "duplicate_removal_report.txt"
if not DEDUP_REPORT.exists():
    _legacy_dedup = PROJECT_ROOT / "duplicate_removal_report.txt"
    if _legacy_dedup.exists():
        DEDUP_REPORT = _legacy_dedup
