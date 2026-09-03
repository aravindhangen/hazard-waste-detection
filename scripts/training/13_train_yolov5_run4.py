"""
Run 4 — YOLOv5s-Seg academic baseline / production model.

Trains YOLOv5s-seg on the frozen hazard_dataset_clean split using the official
ultralytics/yolov5 repository, evaluates on val/test, and writes reports under
runs/yolov5s_run4/ only.
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
import re
import shutil
import subprocess
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from hazard_detection.config import (
    CLASS_NAMES,
    CLEAN_DATASET_ROOT,
    CLEAN_DATA_YAML,
    PROJECT_ROOT,
    RUN4_DIR,
    RUN4_PRETRAINED,
    YOLOV5_DIR,
)

DATA_YAML = CLEAN_DATA_YAML
RUN4_ROOT = RUN4_DIR
RUN4_TRAIN_DIR = RUN4_ROOT / "train"
RUN4_WEIGHTS_DIR = RUN4_ROOT / "weights"
RUN4_EVAL_DIR = RUN4_ROOT / "evaluation"
BEST_RUN4_NAME = "best_yolov5s.pt"
DEFAULT_RUN_NAME = "train"
DEFAULT_CFG = "models/segment/yolov5s-seg.yaml"
DEFAULT_HYP = "data/hyps/hyp.scratch-high.yaml"
PRETRAINED_NAME = "yolov5s-seg.pt"
PRETRAINED_URL = (
    "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s-seg.pt"
)


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


def _safe_print(text: str) -> None:
  try:
    print(text)
  except UnicodeEncodeError:
    print(text.encode("ascii", errors="replace").decode("ascii"))


def run_command(command: list[str], cwd: Path, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("\n$ " + " ".join(command))
    if capture:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.stdout:
            _safe_print(result.stdout)
        if result.stderr:
            _safe_print(result.stderr)
    else:
        result = subprocess.run(command, cwd=str(cwd), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")
    return result


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


def ensure_yolov5_repo() -> None:
    if YOLOV5_DIR.exists():
        return
    run_command(
        ["git", "clone", "--depth", "1", "https://github.com/ultralytics/yolov5.git", "yolov5"],
        PROJECT_ROOT,
    )


def ensure_pretrained_weights() -> Path:
    if RUN4_PRETRAINED.exists():
        return RUN4_PRETRAINED

    target = YOLOV5_DIR / PRETRAINED_NAME
    if target.exists():
        return target

    print(f"Downloading pretrained weights: {PRETRAINED_URL}")
    urllib.request.urlretrieve(PRETRAINED_URL, target)
    return target


def train_run4(
    device: str,
    epochs: int,
    batch_size: int,
    img_size: int,
    workers: int,
    resume: bool,
) -> Path:
    ensure_yolov5_repo()
    ensure_run4_layout()

    last_ckpt = RUN4_TRAIN_DIR / "weights" / "last.pt"
    if resume and last_ckpt.exists():
        weights = last_ckpt
        print(f"Resuming from checkpoint: {weights}")
    else:
        weights = ensure_pretrained_weights()

    command = [
        sys.executable,
        "segment/train.py",
        "--workers",
        str(workers),
        "--device",
        device,
        "--batch-size",
        str(batch_size),
        "--data",
        str(DATA_YAML.resolve()),
        "--imgsz",
        str(img_size),
        "--cfg",
        DEFAULT_CFG,
        "--weights",
        str(weights.resolve()),
        "--name",
        DEFAULT_RUN_NAME,
        "--hyp",
        DEFAULT_HYP,
        "--epochs",
        str(epochs),
        "--exist-ok",
        "--patience",
        "50",
        "--project",
        str(RUN4_ROOT.resolve()),
    ]
    if resume and last_ckpt.exists():
        command.extend(["--resume", str(last_ckpt.resolve())])

    print("\nStarting Run 4 training (YOLOv5s-Seg)...")
    print(f"  Output directory: {RUN4_ROOT}")
    run_command(command, YOLOV5_DIR)

    source_best = RUN4_TRAIN_DIR / "weights" / "best.pt"
    if not source_best.exists():
        raise FileNotFoundError(f"Run 4 best weights not found: {source_best}")

    target_best = RUN4_WEIGHTS_DIR / BEST_RUN4_NAME
    shutil.copy2(source_best, target_best)
    print(f"\nCopied Run 4 checkpoint to: {target_best}")
    return target_best


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def parse_metrics_output(output: str, class_names: list[str]) -> SplitMetrics:
    metrics = SplitMetrics(split="")
    lines = _strip_ansi(output).splitlines()

    overall_pattern = re.compile(
        r"^all\s+(\d+)\s+(\d+)\s+"
        r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
        r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
    )
    for line in lines:
        match = overall_pattern.match(line.strip())
        if match:
            metrics.precision_mask = float(match.group(7))
            metrics.recall_mask = float(match.group(8))
            metrics.map50_mask = float(match.group(9))
            metrics.map_mask = float(match.group(10))
            metrics.f1_mask = f1_score(metrics.precision_mask, metrics.recall_mask)
            break

    class_pattern = re.compile(
        r"^(\S+)\s+\d+\s+\d+\s+"
        r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
        r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
    )
    for line in lines:
        match = class_pattern.match(line.strip())
        if not match:
            continue
        class_name = match.group(1)
        if class_name not in class_names:
            continue
        precision = float(match.group(6))
        recall = float(match.group(7))
        metrics.per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1_score(precision, recall),
            "map50": float(match.group(8)),
            "map50_95": float(match.group(9)),
        }

    speed_match = re.search(
        r"Speed: [\d.]+ms pre-process, ([\d.]+)ms inference, ([\d.]+)ms NMS per image",
        output,
    )
    if speed_match:
        metrics.inference_ms = float(speed_match.group(1)) + float(speed_match.group(2))

    return metrics


def evaluate_split(
    weights: Path,
    split: str,
    device: str,
    batch_size: int,
    img_size: int,
) -> SplitMetrics:
    class_names = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]
    command = [
        sys.executable,
        "segment/val.py",
        "--data",
        str(DATA_YAML.resolve()),
        "--weights",
        str(weights.resolve()),
        "--batch-size",
        str(batch_size),
        "--imgsz",
        str(img_size),
        "--task",
        split,
        "--device",
        device,
        "--verbose",
        "--project",
        str(RUN4_EVAL_DIR.resolve()),
        "--name",
        split,
        "--exist-ok",
    ]
    result = run_command(command, YOLOV5_DIR, capture=True)
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    metrics = parse_metrics_output(output, class_names)
    metrics.split = split
    return metrics


def evaluate_run4(
    weights: Path,
    device: str,
    batch_size: int,
    img_size: int,
) -> tuple[SplitMetrics, SplitMetrics, float | None]:
    print("\nRunning Run 4 validation evaluation...")
    val_metrics = evaluate_split(weights, "val", device, batch_size, img_size)

    print("\nRunning Run 4 untouched test evaluation...")
    test_metrics = evaluate_split(weights, "test", device, batch_size, img_size)

    print("\nBenchmarking Run 4 inference speed (batch=1)...")
    speed_command = [
        sys.executable,
        "segment/val.py",
        "--data",
        str(DATA_YAML.resolve()),
        "--weights",
        str(weights.resolve()),
        "--batch-size",
        "1",
        "--imgsz",
        str(img_size),
        "--task",
        "speed",
        "--device",
        device,
        "--project",
        str(RUN4_EVAL_DIR.resolve()),
        "--name",
        "speed",
        "--exist-ok",
    ]
    speed_result = run_command(speed_command, YOLOV5_DIR, capture=True)
    speed_output = (speed_result.stdout or "") + "\n" + (speed_result.stderr or "")
    speed_match = re.search(
        r"Speed: [\d.]+ms pre-process, ([\d.]+)ms inference, ([\d.]+)ms NMS per image",
        speed_output,
    )
    infer_ms = None
    if speed_match:
        infer_ms = float(speed_match.group(1)) + float(speed_match.group(2))
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
        "NOTE: Run 4 uses the official ultralytics/yolov5 segmentation repo.",
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
    parser.add_argument("--batch-size", type=int, default=4)
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

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset config not found: {DATA_YAML}")

    ensure_yolov5_repo()

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
