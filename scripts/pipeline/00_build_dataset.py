"""Merge source Roboflow exports into hazard_dataset with a 70/20/10 split."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


import random
import shutil
from pathlib import Path

import yaml

from hazard_detection.config import (
    CLASS_NAMES,
    DATASET_ROOT,
    RANDOM_STATE,
    SOURCE_DATASETS,
    SPLITS,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)


def copy_with_prefix(src_images, src_labels, pool_images, pool_labels, prefix):
    copied = 0
    for image_path in sorted(src_images.iterdir()):
        if not image_path.is_file():
            continue
        label_path = src_labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            print(f"WARNING: missing label for {image_path.name}")
            continue
        new_stem = f"{prefix}_{image_path.stem}"
        shutil.copy2(image_path, pool_images / f"{new_stem}{image_path.suffix.lower()}")
        shutil.copy2(label_path, pool_labels / f"{new_stem}.txt")
        copied += 1
    return copied


def main():
    pool_root = DATASET_ROOT / "initial_pool"
    pool_images = pool_root / "images"
    pool_labels = pool_root / "labels"

    if DATASET_ROOT.exists():
        shutil.rmtree(DATASET_ROOT)

    for split in SPLITS:
        (DATASET_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    pool_images.mkdir(parents=True, exist_ok=True)
    pool_labels.mkdir(parents=True, exist_ok=True)

    print("Consolidating source datasets into initial pool...")
    total = 0
    for source in SOURCE_DATASETS:
        root = source["path"]
        prefix = source["prefix"]
        src_images = root / "train" / "images"
        src_labels = root / "train" / "labels"
        if not src_images.exists():
            raise FileNotFoundError(f"Missing images directory: {src_images}")
        count = copy_with_prefix(src_images, src_labels, pool_images, pool_labels, prefix)
        print(f"  {prefix}: copied {count} image/label pairs")
        total += count

    all_images = sorted(pool_images.iterdir())
    print(f"\nTotal images in pool: {len(all_images)}")
    if not all_images:
        raise RuntimeError("No images found in source datasets.")

    rng = random.Random(RANDOM_STATE)
    shuffled = all_images[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = round(n * TEST_RATIO)
    n_val = round(n * VAL_RATIO)
    n_train = n - n_test - n_val

    train_images = shuffled[:n_train]
    val_images = shuffled[n_train : n_train + n_val]
    test_images = shuffled[n_train + n_val :]

    def move_to_split(image_list, split):
        target_images = DATASET_ROOT / "images" / split
        target_labels = DATASET_ROOT / "labels" / split
        for image_path in image_list:
            label_path = pool_labels / f"{image_path.stem}.txt"
            shutil.move(str(image_path), target_images / image_path.name)
            if label_path.exists():
                shutil.move(str(label_path), target_labels / label_path.name)
            else:
                print(f"WARNING: no label for {image_path.name}")

    print("\nSplitting 70/20/10...")
    move_to_split(train_images, "train")
    move_to_split(val_images, "val")
    move_to_split(test_images, "test")

    shutil.rmtree(pool_root)

    data_yaml = {
        "path": str(DATASET_ROOT.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "names": [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)],
    }
    yaml_path = DATASET_ROOT / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    print(f"\nWrote {yaml_path}")
    print("\nFinal counts:")
    for split in SPLITS:
        img_n = len(list((DATASET_ROOT / "images" / split).iterdir()))
        lbl_n = len(list((DATASET_ROOT / "labels" / split).glob("*.txt")))
        print(f"  {split}: {img_n} images, {lbl_n} labels")

    print(f"\nDataset ready at: {DATASET_ROOT}")


if __name__ == "__main__":
    main()
