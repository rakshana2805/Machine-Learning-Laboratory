
from pathlib import Path
from collections import defaultdict
import hashlib

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
SPLITS = ["train", "val", "test"]

hash_locations = defaultdict(list)

for split in SPLITS:
    split_dir = DATASET / split

    if not split_dir.exists():
        print(f"Missing folder: {split_dir}")
        continue

    for path in split_dir.rglob("*"):
        if not path.is_file():
            continue

        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hash_locations[digest].append(
                (split, str(path.relative_to(DATASET)))
            )
        except OSError as error:
            print(f"Cannot read {path}: {error}")

cross_split_duplicates = []

for digest, locations in hash_locations.items():
    splits_found = {split for split, _ in locations}

    if len(splits_found) > 1:
        cross_split_duplicates.append(locations)

print("\n========== DUPLICATE CHECK ==========")

if not cross_split_duplicates:
    print("No exact duplicate files found across splits.")
else:
    print(
        "Exact duplicate groups across splits:",
        len(cross_split_duplicates)
    )

    for group in cross_split_duplicates[:30]:
        print("\nDuplicate group:")
        for split, filename in group:
            print(f"  {split}: {filename}")

print("\nCheck completed. No files were changed.")
