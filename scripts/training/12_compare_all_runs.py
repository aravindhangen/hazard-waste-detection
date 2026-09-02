"""Compare Run 4 (YOLOv5s), Run 2 (YOLO11s), and Run 3 (YOLOv8s) on held-out test metrics."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
from dataclasses import dataclass
from datetime import datetime

from hazard_detection.config import COMPARISON_DIR, RUN2_REPORT_JSON, RUN3_REPORT_JSON, RUN4_REPORT_JSON

OUTPUT_DIR = COMPARISON_DIR
OUTPUT_TXT = OUTPUT_DIR / "all_runs_comparison.txt"
OUTPUT_JSON = OUTPUT_DIR / "all_runs_comparison.json"
MEANINGFUL_DELTA = 0.03


@dataclass
class RunSnapshot:
    run_id: str
    name: str
    model: str
    cylinder_recall: float
    map50: float
    map50_95: float
    recall: float
    precision: float
    fps: float | None
    weights: str
    available: bool = True


def _load_report(path: Path, run_id: str, label: str, default_model: str, default_weights: str) -> RunSnapshot | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    test = data["test"]
    cylinder = test["per_class"].get("Cylinder", {})
    infer_ms = data.get("inference_ms")
    return RunSnapshot(
        run_id=run_id,
        name=label,
        model=data.get("model", default_model),
        cylinder_recall=float(cylinder.get("recall", 0.0)),
        map50=float(test["map50_mask"]),
        map50_95=float(test["map_mask"]),
        recall=float(test["recall_mask"]),
        precision=float(test["precision_mask"]),
        fps=(1000.0 / infer_ms) if infer_ms else None,
        weights=data.get("weights", default_weights),
    )


def fmt(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def main() -> None:
    runs: list[RunSnapshot] = []

    run4 = _load_report(
        RUN4_REPORT_JSON,
        "run4",
        "Run 4 (YOLOv5s)",
        "YOLOv5s-Seg",
        "runs/yolov5s_run4/weights/best_yolov5s.pt",
    )
    if run4:
        runs.append(run4)

    run2 = _load_report(
        RUN2_REPORT_JSON,
        "run2",
        "Run 2 (YOLO11s)",
        "YOLO11s-Seg",
        "runs/yolo11s_run2/weights/best_yolo11s.pt",
    )
    if run2:
        runs.append(run2)

    run3 = _load_report(
        RUN3_REPORT_JSON,
        "run3",
        "Run 3 (YOLOv8s)",
        "YOLOv8s-Seg",
        "runs/yolov8s_run3/weights/best_yolov8s.pt",
    )
    if run3:
        runs.append(run3)

    if not runs:
        raise FileNotFoundError(
            "No run reports found. Train at least one model, e.g. "
            "python scripts/training/13_train_yolov5_run4.py --device 0"
        )

    best_cylinder = max(runs, key=lambda r: r.cylinder_recall)
    best_map50 = max(runs, key=lambda r: r.map50)

    lines = [
        "Hazard Waste Detection — All Runs Comparison",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Models: YOLOv5s (Run 4), YOLO11s (Run 2), YOLOv8s (Run 3)",
        "Priority: Cylinder Recall > mAP@50 > mAP@50:95 > FPS",
        f"Meaningful delta threshold: {MEANINGFUL_DELTA:.2f} on 39-image test set",
        "",
        f"{'Run':<22} {'Model':<18} {'Cyl.Recall':>10} {'mAP@50':>8} {'Recall':>8} {'FPS':>8}",
        "-" * 78,
    ]

    for run in runs:
        lines.append(
            f"{run.name:<22} {run.model:<18} {fmt(run.cylinder_recall):>10} {fmt(run.map50):>8} "
            f"{fmt(run.recall):>8} {fmt(run.fps):>8}"
        )

    if not run4:
        lines.extend(
            [
                "",
                "Run 4 (YOLOv5s) not available yet.",
                "Train with: python scripts/training/13_train_yolov5_run4.py --device 0",
            ]
        )

    lines.extend(
        [
            "",
            "Leaders on held-out test set",
            f"  Best Cylinder recall: {best_cylinder.model} ({best_cylinder.cylinder_recall:.3f})",
            f"  Best mAP@50:          {best_map50.model} ({best_map50.map50:.3f})",
            "",
            "Production recommendation",
            "  YOLOv5s (Run 4) is the default production model for cloud-friendly deployment.",
            "  Compare against YOLO11s and YOLOv8s using the same frozen test split.",
            "",
            "Production weights:",
            "  runs/yolov5s_run4/weights/best_yolov5s.pt",
        ]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "runs": [run.__dict__ for run in runs],
        "best_cylinder_recall": best_cylinder.__dict__,
        "best_map50": best_map50.__dict__,
        "run4_available": run4 is not None,
        "run3_available": run3 is not None,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    OUTPUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
