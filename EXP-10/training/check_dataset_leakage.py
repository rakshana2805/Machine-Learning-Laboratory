
from pathlib import Path
from collections import defaultdict
import hashlib
import re

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"

SPLITS = ["train", "val", "test"]
CLASSES = ["Normal", "Glaucoma"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Hamming distance <= 5 indicates a possible near-duplicate.
# It is a candidate for manual review, not proof of duplication.
MAX_DISTANCE = 5


def image_files():
    records = []

    for split in SPLITS:
        for class_name in CLASSES:
            folder = DATASET / split / class_name

            if not folder.exists():
                print(f"WARNING: Missing folder: {folder}")
                continue

            for path in sorted(folder.iterdir()):
                if path.is_file() and path.suffix.lower() in EXTENSIONS:
                    records.append({
                        "path": path,
                        "split": split,
                        "class_name": class_name
                    })

    return records


def exact_hash(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def difference_hash(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None

    resized = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    differences = resized[:, 1:] > resized[:, :-1]

    value = 0
    for bit in differences.flatten():
        value = (value << 1) | int(bit)

    return value


def normalized_name(path):
    # Removes common augmentation prefixes such as aug_blur_100_
    name = path.stem.lower()
    name = re.sub(r"^aug_[a-z0-9]+_\d+_", "", name)
    return name


def main():
    records = image_files()

    print(f"Total images scanned: {len(records)}")

    if not records:
        print("No images found. Check your dataset folders.")
        return

    for record in records:
        try:
            record["sha256"] = exact_hash(record["path"])
            record["dhash"] = difference_hash(record["path"])
            record["normalized_name"] = normalized_name(record["path"])
        except Exception as error:
            print(f"Could not process {record['path']}: {error}")
            record["sha256"] = None
            record["dhash"] = None

    # 1. Exact byte-for-byte duplicates across different splits
    exact_groups = defaultdict(list)

    for record in records:
        if record["sha256"]:
            exact_groups[record["sha256"]].append(record)

    exact_found = 0

    print("\n========== EXACT DUPLICATES ACROSS SPLITS ==========")

    for group in exact_groups.values():
        splits = {item["split"] for item in group}

        if len(group) > 1 and len(splits) > 1:
            exact_found += 1
            print("\nDuplicate group:")
            for item in group:
                print(
                    f"  {item['split']}/{item['class_name']}/"
                    f"{item['path'].name}"
                )

    if exact_found == 0:
        print("No exact cross-split duplicates found.")

    # 2. Same normalized filename across different splits
    name_groups = defaultdict(list)

    for record in records:
        name_groups[record["normalized_name"]].append(record)

    name_found = 0

    print("\n========== POSSIBLE MATCHING FILENAMES ==========")

    for name, group in name_groups.items():
        splits = {item["split"] for item in group}

        if name and len(group) > 1 and len(splits) > 1:
            name_found += 1
            print(f"\nName key: {name}")

            for item in group:
                print(
                    f"  {item['split']}/{item['class_name']}/"
                    f"{item['path'].name}"
                )

    if name_found == 0:
        print("No matching normalized filenames found.")

    # 3. Perceptual near-duplicates across different splits
    valid = [r for r in records if r["dhash"] is not None]
    train_records = [r for r in valid if r["split"] == "train"]
    other_records = [r for r in valid if r["split"] in ("val", "test")]

    near_found = 0
    shown = 0
    MAX_TO_DISPLAY = 100

    print("\n========== POSSIBLE VISUAL NEAR-DUPLICATES ==========")
    print(f"Checking train images against val/test images.")
    print(f"Threshold: Hamming distance <= {MAX_DISTANCE}")

    for train_item in train_records:
        for other_item in other_records:
            distance = (
                train_item["dhash"] ^ other_item["dhash"]
            ).bit_count()

            if distance <= MAX_DISTANCE:
                near_found += 1

                if shown < MAX_TO_DISPLAY:
                    print(
                        f"\nDistance: {distance}\n"
                        f"  train/{train_item['class_name']}/"
                        f"{train_item['path'].name}\n"
                        f"  {other_item['split']}/"
                        f"{other_item['class_name']}/"
                        f"{other_item['path'].name}"
                    )
                    shown += 1

    if near_found == 0:
        print("No near-duplicate candidates at this threshold.")
    elif near_found > MAX_TO_DISPLAY:
        print(
            f"\nOnly first {MAX_TO_DISPLAY} of "
            f"{near_found} candidate pairs displayed."
        )

    print("\n========== SUMMARY ==========")
    print(f"Images scanned: {len(records)}")
    print(f"Exact duplicate groups crossing splits: {exact_found}")
    print(f"Matching filename groups: {name_found}")
    print(f"Near-duplicate candidate pairs: {near_found}")
    print("\nNo files were modified or deleted.")
    print("Review candidate pairs before deciding whether leakage exists.")


if __name__ == "__main__":
    main()
