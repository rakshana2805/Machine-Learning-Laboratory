
from pathlib import Path

from backend.ml.preprocessing import preprocess_image
from backend.ml.feature_extraction import extract_features

image_folder = Path("dataset/train/Normal")

images = [
    p for p in image_folder.iterdir()
    if p.suffix.lower() in [
        ".jpg", ".jpeg", ".png",
        ".bmp", ".tif", ".tiff"
    ]
]

if not images:
    raise FileNotFoundError(
        f"No images found in {image_folder}"
    )

image_path = images[0]

print("Testing image:", image_path)

img = preprocess_image(image_path)
features = extract_features(img)

print("Image shape:", img.shape)
print("Feature shape:", features.shape)
print("Feature count:", len(features))
print("Feature extraction successful!")
