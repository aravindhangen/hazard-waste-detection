import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pathlib import Path

from hazard_detection.config import CLASS_NAMES, LABEL_DIR, SPLITS


def validate_polygon_labels(label_dir):
    label_dir = Path(label_dir)
    errors = []

    for label_file in sorted(label_dir.glob("*.txt")):
        with open(label_file, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()

                if len(parts) < 7:
                    errors.append(
                        f"{label_file.name}:{line_number} Too few polygon coordinates"
                    )
                    continue

                try:
                    class_id = int(parts[0])
                except ValueError:
                    errors.append(
                        f"{label_file.name}:{line_number} Invalid class ID"
                    )
                    continue

                if class_id not in CLASS_NAMES:
                    errors.append(
                        f"{label_file.name}:{line_number} Invalid class ID: {class_id}"
                    )

                try:
                    coordinates = [float(x) for x in parts[1:]]
                except ValueError:
                    errors.append(
                        f"{label_file.name}:{line_number} Non-numeric coordinate"
                    )
                    continue

                if len(coordinates) % 2 != 0:
                    errors.append(
                        f"{label_file.name}:{line_number} Odd number of coordinates"
                    )
                    continue

                if len(coordinates) < 6:
                    errors.append(
                        f"{label_file.name}:{line_number} Polygon has fewer than 3 points"
                    )

                for index, value in enumerate(coordinates):
                    if not 0 <= value <= 1:
                        coordinate_type = "x" if index % 2 == 0 else "y"
                        errors.append(
                            f"{label_file.name}:{line_number} "
                            f"{coordinate_type} coordinate {value} outside [0,1]"
                        )

    return errors


def main():
    all_errors = []

    for split in SPLITS:
        print("\n" + "=" * 60)
        print(f"VALIDATING: {split.upper()}")
        print("=" * 60)

        errors = validate_polygon_labels(LABEL_DIR / split)
        all_errors.extend((split, error) for error in errors)

        if not errors:
            print("OK - No annotation errors found")
        else:
            print(f"FAIL - {len(errors)} errors found")
            for error in errors[:50]:
                print(" ", error)

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print("Total errors:", len(all_errors))

    if not all_errors:
        print("OK - ALL POLYGON ANNOTATIONS ARE VALID")
    else:
        print("FAIL - Annotation problems must be fixed before training")


if __name__ == "__main__":
    main()
