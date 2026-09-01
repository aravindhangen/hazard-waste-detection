"""YOLO segmentation label validation and cleaning."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from hazard_detection.config import CLASS_NAMES


def validate_label_line(parts: list[str]) -> tuple[bool, str | None]:
    if len(parts) < 7:
        return False, "too_few_coordinates"

    try:
        class_id = int(parts[0])
    except ValueError:
        return False, "invalid_class_id"

    if class_id not in CLASS_NAMES:
        return False, "unknown_class_id"

    try:
        coordinates = [float(value) for value in parts[1:]]
    except ValueError:
        return False, "non_numeric_coordinate"

    if len(coordinates) % 2 != 0:
        return False, "odd_coordinate_count"

    if len(coordinates) < 6:
        return False, "fewer_than_three_points"

    for index, value in enumerate(coordinates):
        if not 0 <= value <= 1:
            axis = "x" if index % 2 == 0 else "y"
            return False, f"{axis}_outside_range"

    return True, None


def clean_label_text(text: str) -> tuple[list[str], list[str]]:
    valid_lines: list[str] = []
    removed: list[str] = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        is_valid, reason = validate_label_line(parts)
        if is_valid:
            valid_lines.append(stripped)
        else:
            removed.append(f"line {line_number}: {reason} -> {stripped[:80]}")

    return valid_lines, removed


def score_label_file(label_path: Path) -> tuple[int, int, Counter]:
    if not label_path.exists():
        return 0, 0, Counter()

    valid_lines, removed = clean_label_text(label_path.read_text(encoding="utf-8"))
    class_counts = Counter()
    for line in valid_lines:
        class_counts[int(line.split()[0])] += 1

    return len(valid_lines), len(removed), class_counts


def primary_class(class_counts: Counter) -> int | None:
    if not class_counts:
        return None
    return max(class_counts.items(), key=lambda item: (item[1], -item[0]))[0]


def validate_polygon_labels(label_dir: Path) -> list[str]:
    errors: list[str] = []

    for label_file in sorted(label_dir.glob("*.txt")):
        for line_number, line in enumerate(
            label_file.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.strip()
            if not stripped:
                continue

            is_valid, reason = validate_label_line(stripped.split())
            if not is_valid:
                errors.append(f"{label_file.name}:{line_number} {reason}")

    return errors
