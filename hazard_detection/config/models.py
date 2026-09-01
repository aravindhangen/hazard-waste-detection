"""Model catalog with benchmark metrics and weight paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from hazard_detection.config.paths import (
    COMPARISON_DIR,
    RUN1_REPORT_JSON,
    RUN1_WEIGHTS,
    RUN2_REPORT_JSON,
    RUN2_WEIGHTS,
    RUN2_DIR,
    RUN3_REPORT_JSON,
    RUN3_WEIGHTS,
    RUN3_DIR,
)

BackendType = Literal["yolov9", "ultralytics"]

# Run 3 paths (also in hazard_detection.config.paths)


@dataclass(frozen=True)
class BenchmarkMetrics:
    cylinder_recall: float | None = None
    map50: float | None = None
    map50_95: float | None = None
    recall: float | None = None
    precision: float | None = None
    f1: float | None = None
    fps: float | None = None


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    short_name: str
    backend: BackendType
    weights: Path
    role: Literal["production", "experimental", "candidate"]
    description: str
    benchmark: BenchmarkMetrics
    inference_available: bool = True
    benchmark_note: str | None = None
    badge: str | None = None


def _load_run1_benchmark() -> BenchmarkMetrics:
    if not RUN1_REPORT_JSON.exists():
        return BenchmarkMetrics(
            cylinder_recall=0.734,
            map50=0.732,
            map50_95=0.510,
            recall=0.722,
            precision=0.743,
            f1=0.732,
            fps=22.9,
        )
    import json

    data = json.loads(RUN1_REPORT_JSON.read_text(encoding="utf-8"))
    test = data["test"]
    cylinder = test["per_class"]["Cylinder"]
    infer_ms = data.get("inference_ms")
    return BenchmarkMetrics(
        cylinder_recall=float(cylinder["recall"]),
        map50=float(test["map50_mask"]),
        map50_95=float(test["map_mask"]),
        recall=float(test["recall_mask"]),
        precision=float(test["precision_mask"]),
        f1=float(test["f1_mask"]),
        fps=1000.0 / float(infer_ms) if infer_ms else None,
    )


def _load_run2_benchmark() -> BenchmarkMetrics:
    if not RUN2_REPORT_JSON.exists():
        return BenchmarkMetrics(
            cylinder_recall=0.714,
            map50=0.643,
            map50_95=0.444,
            recall=0.648,
            precision=0.702,
            f1=0.674,
            fps=26.6,
        )
    import json

    data = json.loads(RUN2_REPORT_JSON.read_text(encoding="utf-8"))
    test = data["test"]
    cylinder = test["per_class"]["Cylinder"]
    infer_ms = data.get("inference_ms")
    return BenchmarkMetrics(
        cylinder_recall=float(cylinder["recall"]),
        map50=float(test["map50_mask"]),
        map50_95=float(test["map_mask"]),
        recall=float(test["recall_mask"]),
        precision=float(test["precision_mask"]),
        f1=float(test["f1_mask"]),
        fps=1000.0 / float(infer_ms) if infer_ms else None,
    )


def _load_run3_benchmark() -> BenchmarkMetrics:
    if not RUN3_REPORT_JSON.exists():
        return BenchmarkMetrics()
    import json

    data = json.loads(RUN3_REPORT_JSON.read_text(encoding="utf-8"))
    test = data["test"]
    cylinder = test["per_class"].get("Cylinder", {})
    infer_ms = data.get("inference_ms")
    return BenchmarkMetrics(
        cylinder_recall=float(cylinder.get("recall", 0.0)) if cylinder else None,
        map50=float(test["map50_mask"]),
        map50_95=float(test["map_mask"]),
        recall=float(test["recall_mask"]),
        precision=float(test["precision_mask"]),
        f1=float(test["f1_mask"]),
        fps=1000.0 / float(infer_ms) if infer_ms else None,
    )


def _yolov8s_spec() -> ModelSpec:
    weights = _resolve_run3_weights()
    trained = weights.exists() and RUN3_REPORT_JSON.exists()
    benchmark = _load_run3_benchmark() if trained else _load_run3_benchmark_from_comparison()
    if benchmark.map50 is not None and weights.exists():
        trained = True
    return ModelSpec(
        id="yolov8s",
        name="YOLOv8s-Seg",
        short_name="YOLOv8s",
        backend="ultralytics",
        weights=weights,
        role="experimental" if trained else "candidate",
        description=(
            "Experimentally benchmarked baseline (Run 3)."
            if trained
            else "Considered academic baseline. Not experimentally trained on this dataset."
        ),
        benchmark=benchmark,
        inference_available=weights.exists(),
        benchmark_note=None if trained else "Not yet trained — no project benchmark metrics available.",
        badge="Tested" if trained else "Not Yet Trained",
    )


def _resolve_run3_weights() -> Path:
    if RUN3_WEIGHTS.exists():
        return RUN3_WEIGHTS
    alt = RUN3_DIR / "train" / "weights" / "best.pt"
    return alt if alt.exists() else RUN3_WEIGHTS


def _load_run3_benchmark_from_comparison() -> BenchmarkMetrics:
    comparison_path = COMPARISON_DIR / "all_runs_comparison.json"
    if not comparison_path.exists():
        return BenchmarkMetrics()
    import json

    data = json.loads(comparison_path.read_text(encoding="utf-8"))
    run3 = next((run for run in data.get("runs", []) if run.get("run_id") == "run3"), None)
    if not run3:
        return BenchmarkMetrics()
    return BenchmarkMetrics(
        cylinder_recall=float(run3.get("cylinder_recall", 0.0)),
        map50=float(run3.get("map50", 0.0)),
        map50_95=float(run3.get("map50_95", 0.0)) if run3.get("map50_95") is not None else None,
        recall=float(run3.get("recall", 0.0)) if run3.get("recall") is not None else None,
        precision=float(run3.get("precision", 0.0)) if run3.get("precision") is not None else None,
        fps=float(run3.get("fps", 0.0)) if run3.get("fps") is not None else None,
    )


def get_model_catalog() -> dict[str, ModelSpec]:
    """Build a fresh catalog so Run 3 availability updates without a process restart."""
    return {
        "yolov9": ModelSpec(
            id="yolov9",
            name="YOLOv9 GELAN-C-SEG",
            short_name="YOLOv9",
            backend="yolov9",
            weights=RUN1_WEIGHTS,
            role="production",
            description="Official production model (Run 1).",
            benchmark=_load_run1_benchmark(),
            inference_available=RUN1_WEIGHTS.exists(),
            badge="Production",
        ),
        "yolo11s": ModelSpec(
            id="yolo11s",
            name="YOLO11s-Seg",
            short_name="YOLO11s",
            backend="ultralytics",
            weights=RUN2_WEIGHTS,
            role="experimental",
            description="Experimentally benchmarked challenger (Run 2).",
            benchmark=_load_run2_benchmark(),
            inference_available=RUN2_WEIGHTS.exists(),
            badge="Tested",
        ),
        "yolov8s": _yolov8s_spec(),
    }


def get_model_spec(model_id: str) -> ModelSpec:
    catalog = get_model_catalog()
    if model_id not in catalog:
        raise KeyError(f"Unknown model id: {model_id}")
    return catalog[model_id]


MODEL_CATALOG = get_model_catalog()

COMPARISON_REPORT = COMPARISON_DIR / "all_runs_comparison.json"
LEGACY_COMPARISON_REPORT = COMPARISON_DIR / "run1_vs_run2_comparison.json"

DEFAULT_MODEL_ID = "yolov9"
COMPARE_MODEL_ORDER = ("yolov9", "yolo11s", "yolov8s")
