import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from collections import Counter
from pathlib import Path

from hazard_detection.config import CLASS_NAMES, DATASET_ROOT, IMAGE_DIR, IMAGE_EXTENSIONS, LABEL_DIR, SPLITS


def get_images(directory):
    return [
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def audit_split(split):
    image_dir = IMAGE_DIR / split
    label_dir = LABEL_DIR / split

    images = get_images(image_dir)
    labels = list(label_dir.glob("*.txt"))

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing_labels = image_stems - label_stems
    orphan_labels = label_stems - image_stems

    class_counts = Counter()

    for label_file in labels:
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
                except ValueError:
                    pass

    print("\n" + "=" * 60)
    print(split.upper())
    print("=" * 60)

    print("Images :", len(images))
    print("Labels :", len(labels))
    print("Missing labels :", len(missing_labels))
    print("Orphan labels  :", len(orphan_labels))

    print("\nClass distribution:")
    for class_id, name in CLASS_NAMES.items():
        print(f"  {class_id}: {name:<20} {class_counts[class_id]}")

    if missing_labels:
        print("\nImages without labels:")
        for item in sorted(missing_labels):
            print(" ", item)

    if orphan_labels:
        print("\nLabels without images:")
        for item in sorted(orphan_labels):
            print(" ", item)


def main():
    if not DATASET_ROOT.exists():
        print(f"Dataset not found at {DATASET_ROOT}")
        print("Run: python scripts/pipeline/00_build_dataset.py")
        return

    print("\nDATASET ROOT:")
    print(DATASET_ROOT)

    for split in SPLITS:
        audit_split(split)


if __name__ == "__main__":
    main()
