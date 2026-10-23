
from pathlib import Path

dataset_path = Path("dataset")
splits = ["train", "val", "test"]
classes = ["Normal", "Glaucoma"]

total = 0

for split in splits:
    print(f"\n{split.upper()} DATASET")

    for class_name in classes:
        folder = dataset_path / split / class_name

        if not folder.exists():
            print(f"Folder not found: {folder}")
            continue

        images = [
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in
            [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
        ]

        print(f"{class_name}: {len(images)} images")
        total += len(images)

print(f"\nTotal images: {total}")
