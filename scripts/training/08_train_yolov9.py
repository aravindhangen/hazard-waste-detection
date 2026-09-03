"""
Legacy YOLOv9 segmentation training pipeline (Run 1 — archived).

Superseded by scripts/training/13_train_yolov5_run4.py for production training.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


import os

# Avoid OpenMP crash when PyTorch + NumPy/SciPy load duplicate libiomp5md.dll (common on Windows).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from hazard_detection.config import (
    CLASS_NAMES,
    CLEAN_DATASET_ROOT,
    CLEAN_DATA_YAML,
    EVALUATION_REPORTS_DIR,
    PROJECT_ROOT,
    RUN1_TRAIN_DIR,
    RUN1_VAL_DIR,
    SPLITS,
    YOLOV9_DIR,
)
from hazard_detection.data.labels import validate_polygon_labels

DATA_YAML = CLEAN_DATA_YAML
REPORT_DIR = EVALUATION_REPORTS_DIR
RUNS_TRAIN_DIR = RUN1_TRAIN_DIR
RUNS_VAL_DIR = RUN1_VAL_DIR

DEFAULT_RUN_NAME = "hazard_waste_seg"
DEFAULT_CFG = "models/segment/gelan-c-seg.yaml"
DEFAULT_HYP = "data/hyps/hyp.scratch-high.yaml"
PRETRAINED_URL = (
    "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c-seg.pt"
)
PRETRAINED_NAME = "gelan-c-seg.pt"


@dataclass
class SplitMetrics:
    split: str
    images: int = 0
    instances: int = 0
    precision_box: float = 0.0
    recall_box: float = 0.0
    map50_box: float = 0.0
    map_box: float = 0.0
    precision_mask: float = 0.0
    recall_mask: float = 0.0
    map50_mask: float = 0.0
    map_mask: float = 0.0
    f1_mask: float = 0.0
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    inference_ms: float | None = None


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def run_command(command: list[str], cwd: Path, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("\n$ " + " ".join(command))
    if capture:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
    else:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}"
        )
    return result


def check_gpu() -> dict:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install training dependencies first:\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124\n"
            "  pip install -r yolov9/requirements.txt"
        ) from exc

    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": "CPU",
        "device_count": 0,
    }
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_count"] = torch.cuda.device_count()
    return info


def verify_dataset() -> dict:
    if not CLEAN_DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Clean dataset not found: {CLEAN_DATASET_ROOT}. Run 05_deduplicate_and_resplit.py"
        )

    stats = {"splits": {}, "total_images": 0, "total_instances": 0, "errors": 0}
    class_totals = {name: 0 for name in CLASS_NAMES.values()}

    for split in SPLITS:
        image_dir = CLEAN_DATASET_ROOT / "images" / split
        label_dir = CLEAN_DATASET_ROOT / "labels" / split
        images = [
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
        errors = validate_polygon_labels(label_dir)
        stats["splits"][split] = {
            "images": len(images),
            "labels": len(list(label_dir.glob("*.txt"))),
            "errors": len(errors),
        }
        stats["total_images"] += len(images)
        stats["errors"] += len(errors)

        for label_file in label_dir.glob("*.txt"):
            for line in label_file.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    class_totals[CLASS_NAMES[class_id]] += 1
                    stats["total_instances"] += 1

    if stats["errors"] > 0:
        raise RuntimeError(
            f"Dataset has {stats['errors']} annotation errors. Fix before training."
        )

    stats["class_totals"] = class_totals
    return stats


def verify_data_yaml() -> dict:
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Missing data.yaml: {DATA_YAML}")

    with open(DATA_YAML, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    expected_names = [CLASS_NAMES[index] for index in sorted(CLASS_NAMES)]
    actual_names = data.get("names", [])
    if actual_names != expected_names:
        raise ValueError(
            f"Unexpected class names in data.yaml.\n"
            f"Expected: {expected_names}\n"
            f"Found:    {actual_names}"
        )

    for split in SPLITS:
        split_path = CLEAN_DATASET_ROOT / data[split]
        if not split_path.exists():
            raise FileNotFoundError(f"Split path missing: {split_path}")

    data["path"] = str(CLEAN_DATASET_ROOT.resolve())
    return data


def ensure_yolov9_repo() -> None:
    if not YOLOV9_DIR.exists():
        run_command(
            ["git", "clone", "--depth", "1", "https://github.com/WongKinYiu/yolov9.git", "yolov9"],
            PROJECT_ROOT,
        )


def ensure_pretrained_weights() -> Path:
    weights_path = YOLOV9_DIR / PRETRAINED_NAME
    if weights_path.exists():
        return weights_path

    import urllib.request

    print(f"Downloading pretrained weights: {PRETRAINED_URL}")
    urllib.request.urlretrieve(PRETRAINED_URL, weights_path)
    return weights_path


def train_model(
    device: str,
    epochs: int,
    batch_size: int,
    img_size: int,
    run_name: str,
    workers: int,
    resume: str | None = None,
) -> Path:
    ensure_yolov9_repo()

    run_dir = RUNS_TRAIN_DIR / run_name
    last_weights = run_dir / "weights" / "last.pt"

    if resume:
        resume_path = Path(resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        weights = resume_path
        print(f"Resuming from: {weights}")
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
        "--img",
        str(img_size),
        "--cfg",
        DEFAULT_CFG,
        "--weights",
        str(weights),
        "--name",
        run_name,
        "--hyp",
        DEFAULT_HYP,
        "--no-overlap",
        "--epochs",
        str(epochs),
        "--close-mosaic",
        "10",
        "--exist-ok",
        "--patience",
        "50",
    ]

    if resume:
        command.extend(["--resume", str(weights)])

    run_command(command, YOLOV9_DIR)

    best_weights = RUNS_TRAIN_DIR / run_name / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Training finished but best weights not found: {best_weights}")
    return best_weights


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def parse_metrics_output(output: str, class_names: list[str]) -> SplitMetrics:
    metrics = SplitMetrics(split="")
    lines = _strip_ansi(output).splitlines()

    overall_pattern = re.compile(
        r"^all\s+(\d+)\s+(\d+)\s+"
        r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
        r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$"
    )
    class_pattern = re.compile(
        r"^(\S+)\s+(\d+)\s+(\d+)\s+"
        r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
        r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$"
    )

    for line in lines:
        line = line.strip()
        if not line or line.startswith("Class") or "%" in line:
            continue

        overall_match = overall_pattern.match(line)
        if overall_match:
            metrics.images = int(overall_match.group(1))
            metrics.instances = int(overall_match.group(2))
            metrics.precision_box = float(overall_match.group(3))
            metrics.recall_box = float(overall_match.group(4))
            metrics.map50_box = float(overall_match.group(5))
            metrics.map_box = float(overall_match.group(6))
            metrics.precision_mask = float(overall_match.group(7))
            metrics.recall_mask = float(overall_match.group(8))
            metrics.map50_mask = float(overall_match.group(9))
            metrics.map_mask = float(overall_match.group(10))
            metrics.f1_mask = f1_score(metrics.precision_mask, metrics.recall_mask)
            continue

        class_match = class_pattern.match(line)
        if not class_match:
            continue

        class_name = class_match.group(1).strip()
        if class_name == "all":
            continue
        if class_name not in class_names and class_name.replace("_", " ") not in [
            n.replace("_", " ") for n in class_names
        ]:
            continue

        p_mask = float(class_match.group(8))
        r_mask = float(class_match.group(9))
        metrics.per_class[class_name] = {
            "precision": p_mask,
            "recall": r_mask,
            "f1": f1_score(p_mask, r_mask),
            "map50": float(class_match.group(10)),
            "map50_95": float(class_match.group(11)),
        }

    speed_match = re.search(
        r"Speed: .* inference, ([\d.]+)ms NMS per image",
        output,
    )
    if not speed_match:
        speed_match = re.search(
            r"Speed: [\d.]+ms pre-process, ([\d.]+)ms inference, ([\d.]+)ms NMS per image",
            output,
        )
        if speed_match:
            infer_ms = float(speed_match.group(1))
            nms_ms = float(speed_match.group(2))
            metrics.inference_ms = infer_ms + nms_ms
    else:
        metrics.inference_ms = float(speed_match.group(1))

    return metrics


def evaluate_split(
    weights: Path,
    split: str,
    device: str,
    batch_size: int,
    img_size: int,
    run_name: str,
    class_names: list[str],
) -> SplitMetrics:
    command = [
        sys.executable,
        "segment/val.py",
        "--data",
        str(DATA_YAML.resolve()),
        "--weights",
        str(weights.resolve()),
        "--batch-size",
        str(batch_size),
        "--img",
        str(img_size),
        "--task",
        split,
        "--device",
        device,
        "--verbose",
        "--project",
        str(RUNS_VAL_DIR),
        "--name",
        f"{run_name}_{split}",
        "--exist-ok",
    ]
    result = run_command(command, YOLOV9_DIR, capture=True)
    metrics = parse_metrics_output(result.stdout + "\n" + (result.stderr or ""), class_names)
    metrics.split = split
    return metrics


def benchmark_inference(
    weights: Path,
    device: str,
    img_size: int,
    run_name: str,
) -> float | None:
    command = [
        sys.executable,
        "segment/val.py",
        "--data",
        str(DATA_YAML.resolve()),
        "--weights",
        str(weights.resolve()),
        "--batch-size",
        "1",
        "--img",
        str(img_size),
        "--task",
        "speed",
        "--device",
        device,
        "--project",
        str(RUNS_VAL_DIR),
        "--name",
        f"{run_name}_speed",
        "--exist-ok",
    ]
    result = run_command(command, YOLOV9_DIR, capture=True)
    speed_match = re.search(
        r"Speed: [\d.]+ms pre-process, ([\d.]+)ms inference, ([\d.]+)ms NMS per image",
        result.stdout + "\n" + (result.stderr or ""),
    )
    if not speed_match:
        return None
    return float(speed_match.group(1)) + float(speed_match.group(2))


def copy_artifacts(run_name: str) -> dict[str, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Path] = {}

    train_run = RUNS_TRAIN_DIR / run_name
    artifact_names = [
        "results.png",
        "confusion_matrix.png",
        "PR_curve.png",
        "F1_curve.png",
        "P_curve.png",
        "R_curve.png",
    ]
    for name in artifact_names:
        source = train_run / name
        if source.exists():
            target = REPORT_DIR / name
            shutil.copy2(source, target)
            copied[name] = target

    val_confusion = RUNS_VAL_DIR / f"{run_name}_test" / "confusion_matrix.png"
    if val_confusion.exists():
        target = REPORT_DIR / "test_confusion_matrix.png"
        shutil.copy2(val_confusion, target)
        copied["test_confusion_matrix.png"] = target

    return copied


def write_report(
    gpu_info: dict,
    dataset_stats: dict,
    val_metrics: SplitMetrics,
    test_metrics: SplitMetrics,
    infer_ms: float | None,
    img_size: int,
    weights: Path,
    artifacts: dict[str, Path],
) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "final_evaluation_report.txt"

    fps = 1000.0 / infer_ms if infer_ms and infer_ms > 0 else 0.0
    lines = [
        "Hazard Waste Detection - YOLOv9 Segmentation Evaluation Report",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Environment",
        f"  PyTorch: {gpu_info['torch_version']}",
        f"  CUDA available: {gpu_info['cuda_available']}",
        f"  Hardware: {gpu_info['device_name']}",
        "",
        "Dataset",
        f"  Root: {CLEAN_DATASET_ROOT}",
        f"  Total unique images: {dataset_stats['total_images']}",
        f"  Total object instances: {dataset_stats['total_instances']}",
    ]
    for split in SPLITS:
        split_stats = dataset_stats["splits"][split]
        lines.append(
            f"  {split.capitalize()}: {split_stats['images']} images"
        )
    for class_name, count in dataset_stats["class_totals"].items():
        lines.append(f"  {class_name} instances: {count}")

    lines.extend(
        [
            "",
            "Model",
            f"  Weights: {weights}",
            f"  Architecture: {DEFAULT_CFG}",
            f"  Image size: {img_size} x {img_size}",
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
    )

    for class_name in [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]:
        class_metrics = test_metrics.per_class.get(class_name)
        if not class_metrics:
            continue
        lines.extend(
            [
                f"  {class_name}:",
                f"    Precision: {class_metrics['precision']:.4f}",
                f"    Recall:    {class_metrics['recall']:.4f}",
                f"    F1:        {class_metrics['f1']:.4f}",
                f"    mAP@50:    {class_metrics['map50']:.4f}",
                f"    mAP@50:95: {class_metrics['map50_95']:.4f}",
            ]
        )

    lines.extend(
        [
            "",
            "Inference performance",
            f"  Image size: {img_size} x {img_size}",
            f"  Average inference time: {infer_ms:.2f} ms" if infer_ms else "  Average inference time: N/A",
            f"  Approximate FPS: {fps:.2f}" if infer_ms else "  Approximate FPS: N/A",
            "",
            "Saved artifacts",
        ]
    )
    for name, path in artifacts.items():
        lines.append(f"  {name}: {path}")

    json_path = REPORT_DIR / "final_evaluation_report.json"
    json_path.write_text(
        json.dumps(
            {
                "gpu": gpu_info,
                "dataset": dataset_stats,
                "validation": val_metrics.__dict__,
                "test": test_metrics.__dict__,
                "inference_ms": infer_ms,
                "artifacts": {key: str(path) for key, path in artifacts.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate YOLOv9 segmentation.")
    parser.add_argument("--device", default="", help="CUDA device id or 'cpu'")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and evaluate existing best.pt weights.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Optional explicit weights path when using --skip-train.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help=(
            "Resume training from a checkpoint (.pt). "
            "Defaults to runs/train-seg/<run-name>/weights/last.pt when set to 'auto'."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("YOLOv9 SEGMENTATION TRAINING + EVALUATION")
    print("=" * 70)

    gpu_info = check_gpu()
    print("\nGPU check:")
    print(f"  PyTorch: {gpu_info['torch_version']}")
    print(f"  CUDA available: {gpu_info['cuda_available']}")
    print(f"  Device: {gpu_info['device_name']}")

    dataset_stats = verify_dataset()
    print("\nDataset check:")
    print(f"  Total images: {dataset_stats['total_images']}")
    print(f"  Total instances: {dataset_stats['total_instances']}")
    for split in SPLITS:
        print(
            f"  {split.capitalize()}: "
            f"{dataset_stats['splits'][split]['images']} images"
        )

    data_yaml = verify_data_yaml()
    class_names = data_yaml["names"]
    print("\ndata.yaml check:")
    print(f"  Classes: {class_names}")

    device = args.device or ("0" if gpu_info["cuda_available"] else "cpu")
    if device == "cpu":
        print("\nWARNING: Training on CPU will be very slow.")

    ensure_yolov9_repo()

    if args.skip_train:
        if args.weights:
            best_weights = Path(args.weights)
        else:
            best_weights = RUNS_TRAIN_DIR / args.run_name / "weights" / "best.pt"
        if not best_weights.exists():
            raise FileNotFoundError(f"Weights not found: {best_weights}")
    else:
        resume_path = None
        if args.resume:
            if args.resume == "auto":
                resume_path = str(RUNS_TRAIN_DIR / args.run_name / "weights" / "last.pt")
            else:
                resume_path = args.resume

        print("\nStarting training...")
        if resume_path:
            print(f"Resume checkpoint: {resume_path}")
        best_weights = train_model(
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            run_name=args.run_name,
            workers=args.workers,
            resume=resume_path,
        )

    print(f"\nBest checkpoint: {best_weights}")

    print("\nRunning validation evaluation...")
    val_metrics = evaluate_split(
        best_weights,
        split="val",
        device=device,
        batch_size=args.batch_size,
        img_size=args.img_size,
        run_name=args.run_name,
        class_names=class_names,
    )

    print("\nRunning untouched test evaluation...")
    test_metrics = evaluate_split(
        best_weights,
        split="test",
        device=device,
        batch_size=args.batch_size,
        img_size=args.img_size,
        run_name=args.run_name,
        class_names=class_names,
    )

    print("\nBenchmarking inference speed...")
    infer_ms = benchmark_inference(
        best_weights,
        device=device,
        img_size=args.img_size,
        run_name=args.run_name,
    )

    artifacts = copy_artifacts(args.run_name)
    report_path = write_report(
        gpu_info=gpu_info,
        dataset_stats=dataset_stats,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        infer_ms=infer_ms,
        img_size=args.img_size,
        weights=best_weights,
        artifacts=artifacts,
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Validation mAP@50 (mask): {val_metrics.map50_mask:.4f}")
    print(f"Test mAP@50 (mask):       {test_metrics.map50_mask:.4f}")
    print(f"Test F1 (mask):           {test_metrics.f1_mask:.4f}")
    if infer_ms:
        print(f"Inference time:           {infer_ms:.2f} ms ({1000/infer_ms:.2f} FPS)")
    print(f"Report saved to:          {report_path}")


if __name__ == "__main__":
    main()
