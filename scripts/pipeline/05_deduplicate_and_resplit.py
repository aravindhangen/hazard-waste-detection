"""
Deduplicate hazard_dataset by perceptual hash, clean malformed annotations,
and write a fresh stratified 70/20/10 split to hazard_dataset_clean.

The original hazard_dataset/ directory is never modified.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import imagehash
import yaml
from PIL import Image

from hazard_detection.config import (
    CLASS_NAMES,
    CLEAN_DATASET_ROOT,
    DATASET_ROOT,
    DEDUP_REPORT,
    IMAGE_EXTENSIONS,
    RANDOM_STATE,
    SPLITS,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from hazard_detection.data.labels import (
    clean_label_text,
    primary_class,
    score_label_file,
    validate_polygon_labels,
)


REPORT_PATH = DEDUP_REPORT


@dataclass
class ImageRecord:
    split: str
    image_path: Path
    label_path: Path
    image_hash: str
    valid_objects: int
    invalid_lines: int
    class_counts: Counter
    image_bytes: int


def collect_records() -> list[ImageRecord]:
    records: list[ImageRecord] = []

    for split in SPLITS:
        image_dir = DATASET_ROOT / "images" / split
        label_dir = DATASET_ROOT / "labels" / split

        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label_path = label_dir / f"{image_path.stem}.txt"
            try:
                with Image.open(image_path) as image:
                    image_hash = str(imagehash.phash(image))
            except Exception as exc:
                raise RuntimeError(f"Could not hash {image_path}: {exc}") from exc

            valid_objects, invalid_lines, class_counts = score_label_file(label_path)
            records.append(
                ImageRecord(
                    split=split,
                    image_path=image_path,
                    label_path=label_path,
                    image_hash=image_hash,
                    valid_objects=valid_objects,
                    invalid_lines=invalid_lines,
                    class_counts=class_counts,
                    image_bytes=image_path.stat().st_size,
                )
            )

    return records


def choose_keeper(group: list[ImageRecord]) -> ImageRecord:
    def sort_key(record: ImageRecord) -> tuple:
        return (
            record.valid_objects,
            -record.invalid_lines,
            record.image_bytes,
            record.image_path.name,
        )

    return max(group, key=sort_key)


def count_cross_split_groups(groups: dict[str, list[ImageRecord]]) -> int:
    cross_split = 0
    for group in groups.values():
        if len(group) > 1 and len({record.split for record in group}) > 1:
            cross_split += 1
    return cross_split


def stratified_split(
    items: list[tuple[Path, Path, int | None]],
    rng: random.Random,
) -> tuple[list, list, list]:
    by_stratum: dict[int, list[tuple[Path, Path, int | None]]] = defaultdict(list)
    for item in items:
        stratum = item[2] if item[2] is not None else -1
        by_stratum[stratum].append(item)

    train: list[tuple[Path, Path, int | None]] = []
    val: list[tuple[Path, Path, int | None]] = []
    test: list[tuple[Path, Path, int | None]] = []

    for stratum_items in by_stratum.values():
        bucket = stratum_items[:]
        rng.shuffle(bucket)

        n = len(bucket)
        n_test = round(n * TEST_RATIO)
        n_val = round(n * VAL_RATIO)
        n_train = max(n - n_test - n_val, 0)

        train.extend(bucket[:n_train])
        val.extend(bucket[n_train : n_train + n_val])
        test.extend(bucket[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_report(lines: list[str]) -> None:
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Source dataset not found: {DATASET_ROOT}. Run scripts/pipeline/00_build_dataset.py first."
        )

    records = collect_records()
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        groups[record.image_hash].append(record)

    duplicate_groups = [group for group in groups.values() if len(group) > 1]
    cross_split_groups = count_cross_split_groups(groups)

    pool_dir = CLEAN_DATASET_ROOT.parent / "_dedup_pool"
    if pool_dir.exists():
        shutil.rmtree(pool_dir)
    pool_images = pool_dir / "images"
    pool_labels = pool_dir / "labels"
    pool_images.mkdir(parents=True)
    pool_labels.mkdir(parents=True)

    report_lines: list[str] = [
        "Hazard Waste Detection - Deduplication and Resplit Report",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Source dataset: {DATASET_ROOT}",
        f"Output dataset: {CLEAN_DATASET_ROOT}",
        "",
        "SUMMARY (before cleanup)",
        f"  Original images: {len(records)}",
        f"  Duplicate groups: {len(duplicate_groups)}",
        f"  Cross-split leakage groups: {cross_split_groups}",
    ]

    malformed_before = sum(record.invalid_lines for record in records)
    report_lines.append(f"  Malformed annotation lines (detected): {malformed_before}")
    report_lines.extend(["", "DUPLICATE GROUP ACTIONS", ""])

    unique_items: list[tuple[Path, Path, int | None]] = []
    excluded_images: list[str] = []
    removed_duplicates = 0
    removed_malformed_lines = 0

    for image_hash, group in sorted(groups.items(), key=lambda item: item[0]):
        keeper = choose_keeper(group)
        valid_lines, removed_lines = clean_label_text(
            keeper.label_path.read_text(encoding="utf-8")
        )
        removed_malformed_lines += len(removed_lines)

        if len(group) > 1:
            report_lines.append(f"Hash {image_hash}:")
            report_lines.append(f"  KEEP [{keeper.split}] {keeper.image_path.name}")
            for record in group:
                if record.image_path == keeper.image_path:
                    continue
                report_lines.append(
                    f"  DROP [{record.split}] {record.image_path.name}"
                )
                removed_duplicates += 1
            report_lines.append("")

        if not valid_lines:
            excluded_images.append(
                f"{keeper.image_path.name} (no valid annotations after cleanup)"
            )
            continue

        pool_image = pool_images / keeper.image_path.name
        pool_label = pool_labels / f"{keeper.image_path.stem}.txt"
        shutil.copy2(keeper.image_path, pool_image)
        pool_label.write_text("\n".join(valid_lines) + "\n", encoding="utf-8")

        stratum = primary_class(Counter(int(line.split()[0]) for line in valid_lines))
        unique_items.append((pool_image, pool_label, stratum))

    if CLEAN_DATASET_ROOT.exists():
        shutil.rmtree(CLEAN_DATASET_ROOT)

    for split in SPLITS:
        (CLEAN_DATASET_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (CLEAN_DATASET_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(RANDOM_STATE)
    train_items, val_items, test_items = stratified_split(unique_items, rng)
    split_map = {"train": train_items, "val": val_items, "test": test_items}

    for split, items in split_map.items():
        for image_path, label_path, _ in items:
            shutil.move(
                str(image_path),
                CLEAN_DATASET_ROOT / "images" / split / image_path.name,
            )
            shutil.move(
                str(label_path),
                CLEAN_DATASET_ROOT / "labels" / split / label_path.name,
            )

    shutil.rmtree(pool_dir)

    data_yaml = {
        "path": str(CLEAN_DATASET_ROOT.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "names": [CLASS_NAMES[index] for index in sorted(CLASS_NAMES)],
    }
    yaml_path = CLEAN_DATASET_ROOT / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data_yaml, handle, sort_keys=False)

    class_counts = Counter()
    split_counts = {split: 0 for split in SPLITS}
    for split in SPLITS:
        split_counts[split] = len(
            list((CLEAN_DATASET_ROOT / "images" / split).iterdir())
        )
        for label_file in (CLEAN_DATASET_ROOT / "labels" / split).glob("*.txt"):
            for line in label_file.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if parts:
                    class_counts[int(parts[0])] += 1

    validation_errors: list[str] = []
    for split in SPLITS:
        validation_errors.extend(
            validate_polygon_labels(CLEAN_DATASET_ROOT / "labels" / split)
        )

    duplicate_hashes = defaultdict(list)
    for split in SPLITS:
        image_dir = CLEAN_DATASET_ROOT / "images" / split
        for image_path in image_dir.iterdir():
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            with Image.open(image_path) as image:
                duplicate_hashes[str(imagehash.phash(image))].append(
                    (split, image_path.name)
                )

    duplicate_groups_after = [
        group for group in duplicate_hashes.values() if len(group) > 1
    ]
    cross_split_after = sum(
        1
        for group in duplicate_groups_after
        if len({split for split, _ in group}) > 1
    )

    report_lines.extend(
        [
            "",
            "CLEANUP RESULTS",
            f"  Unique images kept: {len(unique_items)}",
            f"  Duplicate images dropped: {removed_duplicates}",
            f"  Malformed lines removed: {removed_malformed_lines}",
            f"  Excluded corrupted images: {len(excluded_images)}",
            "",
            "EXCLUDED IMAGES",
        ]
    )
    if excluded_images:
        report_lines.extend(f"  - {item}" for item in excluded_images)
    else:
        report_lines.append("  (none)")

    report_lines.extend(
        [
            "",
            "NEW SPLIT",
            f"  Train: {split_counts['train']}",
            f"  Validation: {split_counts['val']}",
            f"  Test: {split_counts['test']}",
            "",
            "CLASS DISTRIBUTION (object instances)",
        ]
    )
    for class_id, name in CLASS_NAMES.items():
        report_lines.append(f"  {class_id}: {name}: {class_counts[class_id]}")

    report_lines.extend(
        [
            "",
            "POST-CLEAN VALIDATION",
            f"  Malformed labels after cleanup: {len(validation_errors)}",
            f"  Duplicate groups after resplit: {len(duplicate_groups_after)}",
            f"  Cross-split leakage after resplit: {cross_split_after}",
        ]
    )
    write_report(report_lines)

    print("\n" + "=" * 70)
    print("DEDUPLICATION AND RESPLIT COMPLETE")
    print("=" * 70)
    print(f"Original images:                 {len(records)}")
    print(f"Duplicate groups:                {len(duplicate_groups)}")
    print(f"Cross-split leakage (original):  {cross_split_groups}")
    print(f"Malformed annotation lines:      {malformed_before}")
    print()
    print(f"Unique images after dedup:       {len(unique_items)}")
    print(f"Excluded corrupted images:       {len(excluded_images)}")
    print()
    print("New split:")
    print(f"  Train:       {split_counts['train']}")
    print(f"  Validation:  {split_counts['val']}")
    print(f"  Test:        {split_counts['test']}")
    print()
    print("Class distribution:")
    for class_id, name in CLASS_NAMES.items():
        print(f"  {name}: {class_counts[class_id]}")
    print()
    print(f"Duplicate leakage after resplit: {cross_split_after}")
    print(f"Malformed labels after cleanup:  {len(validation_errors)}")
    print(f"Missing labels:                  0 (enforced by pipeline)")
    print()
    print(f"Clean dataset: {CLEAN_DATASET_ROOT}")
    print(f"Report:        {REPORT_PATH}")

    if validation_errors or cross_split_after > 0:
        print("\nWARNING: Post-clean validation found issues.")
        for error in validation_errors[:20]:
            print(" ", error)
    else:
        print("\nOK - Clean dataset passed post-resplit validation checks.")


if __name__ == "__main__":
    main()
