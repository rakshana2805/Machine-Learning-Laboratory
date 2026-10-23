
from pathlib import Path

dataset_path = Path(r"D:\GlaucomaClassifier\dataset")

splits = ["train", "val", "test"]
classes = ["Normal", "Glaucoma"]

total = 0

for split in splits:
    print(f"\n{split.upper()} DATASET")

    for class_name in classes:
        folder = dataset_path / split / class_name

        if not folder.exists():
            print(f"ERROR: Folder not found: {folder}")
            continue

        images = [
            file for file in folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in
            [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
        ]

        count = len(images)
        total += count

        print(f"{class_name}: {count} images")

print(f"\nTotal images found: {total}")

if total > 0:
    print("Dataset check completed.")
else:
    print("No images found. Check your folder structure.")
