"""
Compare Run 1 (YOLOv9 official baseline) vs Run 2 (YOLO11s experimental).

Priority for production recommendation:
  Cylinder Recall > Overall mAP@50 > Shock Absorber mAP@50:95 > FPS

Does not modify any Run 1 artifacts.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from hazard_detection.config import COMPARISON_DIR, RUN1_REPORT_JSON, RUN2_REPORT_JSON

RUN1_REPORT = RUN1_REPORT_JSON
RUN2_REPORT = RUN2_REPORT_JSON
OUTPUT_DIR = COMPARISON_DIR
OUTPUT_TXT = OUTPUT_DIR / "run1_vs_run2_comparison.txt"
OUTPUT_JSON = OUTPUT_DIR / "run1_vs_run2_comparison.json"

# Minimum deltas worth treating as meaningful on a 39-image test set.
MEANINGFUL_DELTA = 0.03


@dataclass
class RunSnapshot:
    name: str
    model: str
    cylinder_recall: float
    map50: float
    map50_95: float
    f1: float
    recall: float
    precision: float
    fps: float | None
    weights: str


def load_run1() -> RunSnapshot:
    data = json.loads(RUN1_REPORT.read_text(encoding="utf-8"))
    test = data["test"]
    cylinder = test["per_class"]["Cylinder"]
    infer_ms = data.get("inference_ms")
    return RunSnapshot(
        name="Run 1 (official)",
        model="YOLOv9 GELAN-C-SEG",
        cylinder_recall=float(cylinder["recall"]),
        map50=float(test["map50_mask"]),
        map50_95=float(test["map_mask"]),
        f1=float(test["f1_mask"]),
        recall=float(test["recall_mask"]),
        precision=float(test["precision_mask"]),
        fps=(1000.0 / infer_ms) if infer_ms else None,
        weights="yolov9/runs/train-seg/hazard_waste_seg/weights/best.pt",
    )


def load_run2() -> RunSnapshot:
    if not RUN2_REPORT.exists():
        raise FileNotFoundError(
            f"Run 2 report not found: {RUN2_REPORT}\n"
            "Train/evaluate Run 2 first:\n"
            "  python 09_train_yolo11_run2.py --device 0"
        )
    data = json.loads(RUN2_REPORT.read_text(encoding="utf-8"))
    test = data["test"]
    cylinder = test["per_class"].get("Cylinder", {})
    infer_ms = data.get("inference_ms")
    return RunSnapshot(
        name="Run 2 (experimental)",
        model=data.get("model", "YOLO11s-Seg"),
        cylinder_recall=float(cylinder.get("recall", 0.0)),
        map50=float(test["map50_mask"]),
        map50_95=float(test["map_mask"]),
        f1=float(test["f1_mask"]),
        recall=float(test["recall_mask"]),
        precision=float(test["precision_mask"]),
        fps=(1000.0 / infer_ms) if infer_ms else None,
        weights=data.get("weights", "runs/yolo11s_run2/weights/best_yolo11s.pt"),
    )


def fmt(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def delta(run2: float, run1: float) -> str:
    diff = run2 - run1
    sign = "+" if diff >= 0 else ""
    note = ""
    if abs(diff) < MEANINGFUL_DELTA:
        note = " (within noise on 39-image test)"
    return f"{sign}{diff:.3f}{note}"


def recommend(run1: RunSnapshot, run2: RunSnapshot) -> tuple[str, list[str]]:
    reasons: list[str] = []

    if run2.cylinder_recall + 1e-9 < run1.cylinder_recall:
        reasons.append(
            f"Cylinder recall decreased ({run1.cylinder_recall:.3f} -> {run2.cylinder_recall:.3f}). "
            "Safety priority favors keeping Run 1."
        )
        return "Keep Run 1 (YOLOv9) as production model.", reasons

    improvements = 0
    if run2.cylinder_recall > run1.cylinder_recall + MEANINGFUL_DELTA:
        improvements += 1
        reasons.append(f"Cylinder recall improved by {run2.cylinder_recall - run1.cylinder_recall:.3f}.")
    if run2.map50 > run1.map50 + MEANINGFUL_DELTA:
        improvements += 1
        reasons.append(f"Overall test mAP@50 improved by {run2.map50 - run1.map50:.3f}.")
    if run2.map50_95 > run1.map50_95 + MEANINGFUL_DELTA:
        improvements += 1
        reasons.append(f"mAP@50:95 improved by {run2.map50_95 - run1.map50_95:.3f}.")
    if run1.fps and run2.fps and run2.fps > run1.fps * 1.1:
        improvements += 1
        reasons.append(f"FPS improved ({run1.fps:.1f} -> {run2.fps:.1f}).")

    if improvements >= 2 and run2.cylinder_recall >= run1.cylinder_recall:
        return (
            "Consider switching production to Run 2 (YOLO11s) after qualitative review.",
            reasons,
        )

    reasons.append(
        "Run 2 does not show a clear, safety-aligned improvement over Run 1 on the held-out test set."
    )
    return "Keep Run 1 (YOLOv9) as production model.", reasons


def main() -> None:
    run1 = load_run1()
    run2 = load_run2()
    recommendation, reasons = recommend(run1, run2)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "Hazard Waste Detection — Run 1 vs Run 2 Comparison",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Comparison priority: Cylinder Recall > mAP@50 > mAP@50:95 > FPS",
        f"Meaningful delta threshold: {MEANINGFUL_DELTA:.2f} on 39-image test set",
        "",
        f"{'Metric':<22} {'Run 1 (YOLOv9)':>14} {'Run 2 (YOLO11s)':>16} {'Delta':>12}",
        "-" * 68,
        f"{'Cylinder Recall':<22} {fmt(run1.cylinder_recall):>14} {fmt(run2.cylinder_recall):>16} {delta(run2.cylinder_recall, run1.cylinder_recall):>12}",
        f"{'Test Recall':<22} {fmt(run1.recall):>14} {fmt(run2.recall):>16} {delta(run2.recall, run1.recall):>12}",
        f"{'Test Precision':<22} {fmt(run1.precision):>14} {fmt(run2.precision):>16} {delta(run2.precision, run1.precision):>12}",
        f"{'Test F1':<22} {fmt(run1.f1):>14} {fmt(run2.f1):>16} {delta(run2.f1, run1.f1):>12}",
        f"{'Test mAP@50':<22} {fmt(run1.map50):>14} {fmt(run2.map50):>16} {delta(run2.map50, run1.map50):>12}",
        f"{'Test mAP@50:95':<22} {fmt(run1.map50_95):>14} {fmt(run2.map50_95):>16} {delta(run2.map50_95, run1.map50_95):>12}",
        f"{'FPS (approx)':<22} {fmt(run1.fps):>14} {fmt(run2.fps):>16} {'':>12}",
        "",
        "Weights",
        f"  Run 1: {run1.weights}",
        f"  Run 2: {run2.weights}",
        "",
        "Recommendation",
        f"  {recommendation}",
    ]
    for reason in reasons:
        lines.append(f"  - {reason}")

    lines.extend(
        [
            "",
            "Untouched official artifacts",
            "  hazard_dataset_clean/",
            "  evaluation_reports/final_evaluation_report.txt",
            "  yolov9/runs/train-seg/hazard_waste_seg/weights/best.pt",
            "  FastAPI + dashboard deployment",
        ]
    )

    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "priority": ["cylinder_recall", "map50", "map50_95", "fps"],
        "meaningful_delta": MEANINGFUL_DELTA,
        "run1": run1.__dict__,
        "run2": run2.__dict__,
        "recommendation": recommendation,
        "reasons": reasons,
    }

    OUTPUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nSaved: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
