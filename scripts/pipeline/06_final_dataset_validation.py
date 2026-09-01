"""Validate the cleaned hazard_dataset_clean before training."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


from collections import Counter
from pathlib import Path

import imagehash
from PIL import Image

from hazard_detection.config import (
    CLASS_NAMES,
    CLEAN_DATASET_ROOT,
    CLEAN_IMAGE_DIR,
    CLEAN_LABEL_DIR,
    IMAGE_EXTENSIONS,
    SPLITS,
)
from hazard_detection.data.labels import validate_polygon_labels


def get_images(directory: Path) -> list[Path]:
    return [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def audit_split(split: str) -> dict:
    image_dir = CLEAN_IMAGE_DIR / split
    label_dir = CLEAN_LABEL_DIR / split

    images = get_images(image_dir)
    labels = list(label_dir.glob("*.txt"))

    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}

    class_counts = Counter()
    for label_file in labels:
        for line in label_file.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if parts:
                class_counts[int(parts[0])] += 1

    return {
        "images": len(images),
        "labels": len(labels),
        "missing_labels": len(image_stems - label_stems),
        "orphan_labels": len(label_stems - image_stems),
        "class_counts": class_counts,
        "annotation_errors": validate_polygon_labels(label_dir),
    }


def duplicate_report() -> tuple[int, int]:
    hashes: dict[str, list[tuple[str, str]]] = {}
    for split in SPLITS:
        for image_path in get_images(CLEAN_IMAGE_DIR / split):
            with Image.open(image_path) as image:
                key = str(imagehash.phash(image))
            hashes.setdefault(key, []).append((split, image_path.name))

    duplicate_groups = [group for group in hashes.values() if len(group) > 1]
    cross_split = sum(
        1
        for group in duplicate_groups
        if len({split for split, _ in group}) > 1
    )
    return len(duplicate_groups), cross_split


def main() -> None:
    if not CLEAN_DATASET_ROOT.exists():
        print(f"Clean dataset not found: {CLEAN_DATASET_ROOT}")
        print("Run: python 05_deduplicate_and_resplit.py")
        return

    print("\nDATASET ROOT:")
    print(CLEAN_DATASET_ROOT)

    total_errors = 0
    total_class_counts = Counter()

    for split in SPLITS:
        stats = audit_split(split)
        total_errors += len(stats["annotation_errors"])
        total_class_counts.update(stats["class_counts"])

        print("\n" + "=" * 60)
        print(split.upper())
        print("=" * 60)
        print("Images :", stats["images"])
        print("Labels :", stats["labels"])
        print("Missing labels :", stats["missing_labels"])
        print("Orphan labels  :", stats["orphan_labels"])
        print("\nClass distribution:")
        for class_id, name in CLASS_NAMES.items():
            print(f"  {class_id}: {name:<20} {stats['class_counts'][class_id]}")

        if stats["annotation_errors"]:
            print(f"\nFAIL - {len(stats['annotation_errors'])} annotation errors")
            for error in stats["annotation_errors"][:20]:
                print(" ", error)
        else:
            print("\nOK - No annotation errors")

    duplicate_groups, cross_split = duplicate_report()

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print("Total annotation errors:", total_errors)
    print("Duplicate groups:", duplicate_groups)
    print("Cross-split leakage groups:", cross_split)
    print("Missing labels:", sum(audit_split(split)["missing_labels"] for split in SPLITS))

    print("\nOverall class distribution:")
    for class_id, name in CLASS_NAMES.items():
        print(f"  {name}: {total_class_counts[class_id]}")

    if total_errors == 0 and duplicate_groups == 0 and cross_split == 0:
        print("\nOK - CLEAN DATASET IS READY FOR TRAINING")
    else:
        print("\nFAIL - Clean dataset is not ready for training")


if __name__ == "__main__":
    main()
