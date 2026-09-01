import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from hazard_detection.config import CLASS_NAMES, IMAGE_DIR, IMAGE_EXTENSIONS, LABEL_DIR, SPLITS


def load_polygon(label_file):
    polygons = []

    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))

            points = []
            for i in range(0, len(coords), 2):
                points.append((coords[i], coords[i + 1]))

            polygons.append((class_id, points))

    return polygons


def visualize_split(split, samples=5):
    image_dir = IMAGE_DIR / split
    label_dir = LABEL_DIR / split

    images = [
        p for p in image_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    random.seed(42)
    selected = random.sample(images, min(samples, len(images)))

    for image_path in selected:
        label_path = label_dir / f"{image_path.stem}.txt"

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not read image: {image_path}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        polygons = load_polygon(label_path) if label_path.exists() else []

        plt.figure(figsize=(12, 8))
        plt.imshow(image)

        for class_id, points in polygons:
            pixel_points = [
                (int(x * width), int(y * height))
                for x, y in points
            ]

            xs = [p[0] for p in pixel_points]
            ys = [p[1] for p in pixel_points]
            xs.append(xs[0])
            ys.append(ys[0])

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

        plt.title(f"{split.upper()} - {image_path.name}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


def main():
    for split in SPLITS:
        print(f"\nShowing {split} annotations...")
        visualize_split(split, samples=5)


if __name__ == "__main__":
    main()
