"""Report perceptual-hash duplicates. Does NOT delete anything."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


from collections import defaultdict

import imagehash
from PIL import Image

from hazard_detection.config import DATASET_ROOT, IMAGE_DIR, IMAGE_EXTENSIONS, SPLITS


def main():
    if not DATASET_ROOT.exists():
        print(f"Dataset not found at {DATASET_ROOT}")
        print("Run: python scripts/pipeline/00_build_dataset.py")
        return

    hashes = defaultdict(list)

    for split in SPLITS:
        image_dir = IMAGE_DIR / split
        print(f"Scanning {split}...")

        for image_path in image_dir.iterdir():
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            try:
                image = Image.open(image_path)
                image_hash = str(imagehash.phash(image))
                hashes[image_hash].append((split, image_path))
            except Exception as e:
                print(f"ERROR: {image_path.name}: {e}")

    duplicate_groups = [
        files for files in hashes.values() if len(files) > 1
    ]

    print("\n" + "=" * 70)
    print("DUPLICATE REPORT")
    print("=" * 70)
    print("Unique image hashes:", len(hashes))
    print("Duplicate groups:", len(duplicate_groups))

    cross_split_groups = 0
    for number, group in enumerate(duplicate_groups, start=1):
        splits_in_group = {split for split, _ in group}
        if len(splits_in_group) > 1:
            cross_split_groups += 1

        print(f"\nDuplicate Group {number}")
        for split, path in group:
            print(f"  [{split.upper()}] {path.name}")

        if len(splits_in_group) > 1:
            print("  ** CROSS-SPLIT LEAKAGE - review before training **")

    print("\n" + "=" * 70)
    print(f"Groups with cross-split leakage: {cross_split_groups}")
    print("No files were modified.")


if __name__ == "__main__":
    main()
