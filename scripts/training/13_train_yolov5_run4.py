"""
Run 4 — YOLOv5s-Seg academic baseline / production model.

Trains YOLOv5s-seg on the frozen hazard_dataset_clean split, evaluates on the
held-out test set, and writes reports under runs/yolov5s_run4/ only.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from hazard_detection.config import (
    CLASS_NAMES,
    CLEAN_DATASET_ROOT,
    CLEAN_DATA_YAML,
    PRETRAINED_DIR,
    RUN4_DIR,
    RUN4_PRETRAINED,
)

DATA_YAML = CLEAN_DATA_YAML
RUN4_ROOT = RUN4_DIR
RUN4_TRAIN_DIR = RUN4_ROOT / "train"
RUN4_WEIGHTS_DIR = RUN4_ROOT / "weights"
RUN4_EVAL_DIR = RUN4_ROOT / "evaluation"
BEST_RUN4_NAME = "best_yolov5s.pt"
PRETRAINED = str(RUN4_PRETRAINED) if RUN4_PRETRAINED.exists() else "yolov5s-seg.pt"


@dataclass
class SplitMetrics:
    split: str
    precision_mask: float = 0.0
    recall_mask: float = 0.0
    f1_mask: float = 0.0
    map50_mask: float = 0.0
    map_mask: float = 0.0
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    inference_ms: float | None = None


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def check_gpu() -> dict:
    import torch

    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": "CPU",
    }
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
    return info


def ensure_run4_layout() -> None:
    RUN4_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    RUN4_EVAL_DIR.mkdir(parents=True, exist_ok=True)


def train_run4(
    device: str,
    epochs: int,
    batch_size: int,
    img_size: int,
    workers: int,
    resume: bool,
) -> Path:
    from ultralytics import YOLO

    ensure_run4_layout()
    model = YOLO(PRETRAINED)
    train_kwargs = {
        "data": str(DATA_YAML.resolve()),
        "epochs": epochs,
        "imgsz": img_size,
        "batch": batch_size,
        "device": device,
        "workers": workers,
        "project": str(RUN4_ROOT),
        "name": "train",
        "exist_ok": True,
        "pretrained": True,
        "verbose": True,
    }
    last_ckpt = RUN4_TRAIN_DIR / "weights" / "last.pt"
    if resume and last_ckpt.exists():
        train_kwargs["resume"] = str(last_ckpt)
        print(f"Resuming from checkpoint: {last_ckpt}")

    print("\nStarting Run 4 training (YOLOv5s-Seg)...")
    print(f"  Output directory: {RUN4_ROOT}")
    model.train(**train_kwargs)

    source_best = RUN4_TRAIN_DIR / "weights" / "best.pt"
    if not source_best.exists():
        raise FileNotFoundError(f"Run 4 best weights not found: {source_best}")

    target_best = RUN4_WEIGHTS_DIR / BEST_RUN4_NAME
    shutil.copy2(source_best, target_best)
    print(f"\nCopied Run 4 checkpoint to: {target_best}")
    return target_best


def extract_metrics(results, split: str) -> SplitMetrics:
    summary = results.summary(decimals=4)
    per_class: dict[str, dict[str, float]] = {}
    for row in summary:
        class_name = row["Class"]
        precision = float(row["Mask-P"])
        recall = float(row["Mask-R"])
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1_score(precision, recall),
            "map50": float(row["mAP50"]),
            "map50_95": float(row["mAP50-95"]),
        }

    return SplitMetrics(
        split=split,
        precision_mask=float(results.seg.mp),
        recall_mask=float(results.seg.mr),
        f1_mask=f1_score(float(results.seg.mp), float(results.seg.mr)),
        map50_mask=float(results.seg.map50),
        map_mask=float(results.seg.map),
        per_class=per_class,
    )


def evaluate_run4(
    weights: Path,
    device: str,
    batch_size: int,
    img_size: int,
) -> tuple[SplitMetrics, SplitMetrics, float | None]:
    from ultralytics import YOLO

    model = YOLO(str(weights))

    print("\nRunning Run 4 validation evaluation...")
    val_results = model.val(
        data=str(DATA_YAML.resolve()),
        split="val",
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project=str(RUN4_EVAL_DIR),
        name="val",
        exist_ok=True,
        verbose=True,
    )
    val_metrics = extract_metrics(val_results, "val")

    print("\nRunning Run 4 untouched test evaluation...")
    test_results = model.val(
        data=str(DATA_YAML.resolve()),
        split="test",
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project=str(RUN4_EVAL_DIR),
        name="test",
        exist_ok=True,
        verbose=True,
    )
    test_metrics = extract_metrics(test_results, "test")

    print("\nBenchmarking Run 4 inference speed (batch=1)...")
    speed_results = model.val(
        data=str(DATA_YAML.resolve()),
        split="val",
        imgsz=img_size,
        batch=1,
        device=device,
        verbose=False,
    )
    speed = speed_results.speed
    infer_ms = speed["preprocess"] + speed["inference"] + speed["postprocess"]
    return val_metrics, test_metrics, infer_ms


def write_run4_report(
    gpu_info: dict,
    weights: Path,
    val_metrics: SplitMetrics,
    test_metrics: SplitMetrics,
    infer_ms: float | None,
    img_size: int,
    epochs: int,
    batch_size: int,
) -> Path:
    fps = 1000.0 / infer_ms if infer_ms and infer_ms > 0 else 0.0
    report_txt = RUN4_EVAL_DIR / "run4_evaluation_report.txt"
    report_json = RUN4_EVAL_DIR / "run4_evaluation_report.json"

    lines = [
        "Hazard Waste Detection - Run 4 YOLOv5s-Seg Evaluation Report",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "NOTE: Run 4 replaces YOLOv9 as the academic baseline / production Ultralytics model.",
        "",
        "Environment",
        f"  PyTorch: {gpu_info['torch_version']}",
        f"  CUDA available: {gpu_info['cuda_available']}",
        f"  Hardware: {gpu_info['device_name']}",
        "",
        "Dataset (frozen)",
        f"  Root: {CLEAN_DATASET_ROOT}",
        f"  Split: 275 train / 79 val / 39 test",
        "",
        "Model",
        "  Architecture: YOLOv5s-Seg",
        f"  Weights: {weights}",
        f"  Image size: {img_size} x {img_size}",
        f"  Epochs: {epochs}",
        f"  Batch size: {batch_size}",
        "",
        "Validation metrics (segmentation masks)",
        f"  Precision: {val_metrics.precision_mask:.4f}",
        f"  Recall:    {val_metrics.recall_mask:.4f}",
        f"  F1:        {val_metrics.f1_mask:.4f}",
        f"  mAP@50:    {val_metrics.map50_mask:.4f}",
        f"  mAP@50:95: {val_metrics.map_mask:.4f}",
        "",
        "Test metrics (segmentation masks)",
        f"  Precision: {test_metrics.precision_mask:.4f}",
        f"  Recall:    {test_metrics.recall_mask:.4f}",
        f"  F1:        {test_metrics.f1_mask:.4f}",
        f"  mAP@50:    {test_metrics.map50_mask:.4f}",
        f"  mAP@50:95: {test_metrics.map_mask:.4f}",
        "",
        "Per-class test metrics (masks)",
    ]

    for class_name in [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]:
        row = test_metrics.per_class.get(class_name)
        if not row:
            continue
        lines.extend(
            [
                f"  {class_name}:",
                f"    Precision: {row['precision']:.4f}",
                f"    Recall:    {row['recall']:.4f}",
                f"    F1:        {row['f1']:.4f}",
                f"    mAP@50:    {row['map50']:.4f}",
                f"    mAP@50:95: {row['map50_95']:.4f}",
            ]
        )

    lines.extend(
        [
            "",
            "Inference performance",
            f"  Average inference time: {infer_ms:.2f} ms" if infer_ms else "  Average inference time: N/A",
            f"  Approximate FPS: {fps:.2f}" if infer_ms else "  Approximate FPS: N/A",
        ]
    )

    payload = {
        "run": "yolov5s_run4",
        "model": "YOLOv5s-Seg",
        "gpu": gpu_info,
        "weights": str(weights),
        "validation": asdict(val_metrics),
        "test": asdict(test_metrics),
        "inference_ms": infer_ms,
        "training": {"epochs": epochs, "batch_size": batch_size, "img_size": img_size},
    }
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_txt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 4 YOLOv5s-Seg training/evaluation (academic baseline / production)."
    )
    parser.add_argument("--device", default="", help="CUDA device id or 'cpu'")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-train", action="store_true", help="Evaluate existing Run 4 weights only.")
    parser.add_argument("--weights", type=str, default="", help="Optional weights path for --skip-train.")
    parser.add_argument("--resume", action="store_true", help="Resume Run 4 training from last.pt.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpu_info = check_gpu()
    device = args.device or ("0" if gpu_info["cuda_available"] else "cpu")

    print("=" * 70)
    print("RUN 4 — YOLOv5s-SEG (ACADEMIC BASELINE / PRODUCTION)")
    print("=" * 70)
    print(f"\nRun 4 output root: {RUN4_ROOT}")
    print(f"Pretrained base: {PRETRAINED}")

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset config not found: {DATA_YAML}")

    if args.skip_train:
        weights = Path(args.weights) if args.weights else RUN4_WEIGHTS_DIR / BEST_RUN4_NAME
        if not weights.exists():
            fallback = RUN4_TRAIN_DIR / "weights" / "best.pt"
            if fallback.exists():
                weights = fallback
            else:
                raise FileNotFoundError(f"Run 4 weights not found: {weights}")
    else:
        weights = train_run4(
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            workers=args.workers,
            resume=args.resume,
        )

    val_metrics, test_metrics, infer_ms = evaluate_run4(
        weights=weights,
        device=device,
        batch_size=args.batch_size,
        img_size=args.img_size,
    )

    report_path = write_run4_report(
        gpu_info=gpu_info,
        weights=weights,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        infer_ms=infer_ms,
        img_size=args.img_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    print("\n" + "=" * 70)
    print("RUN 4 EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Test mAP@50 (mask): {test_metrics.map50_mask:.4f}")
    print(f"Test recall (mask): {test_metrics.recall_mask:.4f}")
    cylinder = test_metrics.per_class.get("Cylinder", {})
    if cylinder:
        print(f"Cylinder recall:    {cylinder.get('recall', 0.0):.4f}")
    if infer_ms:
        print(f"Inference time:     {infer_ms:.2f} ms ({1000/infer_ms:.2f} FPS)")
    print(f"Report saved to:    {report_path}")
    print("\nNext: python scripts/training/12_compare_all_runs.py")


if __name__ == "__main__":
    main()
