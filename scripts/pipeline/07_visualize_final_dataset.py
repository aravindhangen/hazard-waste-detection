"""Visual QA for hazard_dataset_clean."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


import random
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hazard_detection.config import (
    CLASS_NAMES,
    CLEAN_DATASET_ROOT,
    CLEAN_IMAGE_DIR,
    CLEAN_LABEL_DIR,
    IMAGE_EXTENSIONS,
    SPLITS,
    VISUAL_QA_DIR,
)

# Images where malformed polygon objects were removed during cleanup (keepers).
AFFECTED_IMAGES = [
    "scrap_Screenshot-2025-02-16-235353_png_png.rf.8656ae4f84185c117f5e84853b3548bc.jpg",
    "scrap_Screenshot-2025-02-17-050342_png_png.rf.c4851ef83fab51b42074671ba5895231.jpg",
]

OUTPUT_DIR = VISUAL_QA_DIR


def load_polygons(label_file: Path) -> list[tuple[int, list[tuple[float, float]]]]:
    polygons: list[tuple[int, list[tuple[float, float]]]] = []

    for line in label_file.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue

        class_id = int(parts[0])
        coords = list(map(float, parts[1:]))
        points = [(coords[index], coords[index + 1]) for index in range(0, len(coords), 2)]
        polygons.append((class_id, points))

    return polygons


def render_annotation(image_path: Path, label_path: Path, title: str, output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image: {image_path}")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    polygons = load_polygons(label_path) if label_path.exists() else []

    plt.figure(figsize=(12, 8))
    plt.imshow(image)

    for class_id, points in polygons:
        pixel_points = [(int(x * width), int(y * height)) for x, y in points]
        xs = [point[0] for point in pixel_points] + [pixel_points[0][0]]
        ys = [point[1] for point in pixel_points] + [pixel_points[0][1]]
        plt.plot(xs, ys, linewidth=2)

        center_x = sum(xs[:-1]) / len(xs[:-1])
        center_y = sum(ys[:-1]) / len(ys[:-1])
        plt.text(
            center_x,
            center_y,
            CLASS_NAMES.get(class_id, str(class_id)),
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.7),
        )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def visualize_split(split: str, samples: int = 5) -> None:
    image_dir = CLEAN_IMAGE_DIR / split
    label_dir = CLEAN_LABEL_DIR / split

    images = [
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not images:
        print(f"No images found in {split}.")
        return

    by_name = {path.name: path for path in images}
    selected_names = []

    for name in AFFECTED_IMAGES:
        if name in by_name and split == "train":
            selected_names.append(name)

    random.seed(42)
    remaining = [path for path in images if path.name not in selected_names]
    selected_names.extend(
        path.name
        for path in random.sample(remaining, min(samples, len(remaining)))
    )

    for image_name in selected_names:
        image_path = by_name[image_name]
        label_path = label_dir / f"{image_path.stem}.txt"
        tag = " [AFFECTED - malformed object removed]" if image_name in AFFECTED_IMAGES else ""
        output_path = OUTPUT_DIR / split / f"{image_path.stem}{tag.replace(' ', '_').replace('[', '').replace(']', '')}.jpg"
        render_annotation(
            image_path,
            label_path,
            f"{split.upper()} - {image_path.name}{tag}",
            output_path,
        )


def main() -> None:
    if not CLEAN_DATASET_ROOT.exists():
        print(f"Clean dataset not found: {CLEAN_DATASET_ROOT}")
        print("Run: python 05_deduplicate_and_resplit.py")
        return

    print(f"Saving visual QA images to: {OUTPUT_DIR}")
    print("\nAffected images (malformed objects removed) will be included in TRAIN samples:")
    for name in AFFECTED_IMAGES:
        print(f"  - {name}")

    for split in SPLITS:
        print(f"\nGenerating {split} annotation previews...")
        visualize_split(split, samples=5)

    print(f"\nVisual QA complete. Open images in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
