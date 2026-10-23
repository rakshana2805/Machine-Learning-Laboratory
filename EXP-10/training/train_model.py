
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.ml.preprocessing import preprocess_image
from backend.ml.feature_extraction import extract_features

DATASET_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "backend" / "models"
MODEL_PATH = MODEL_DIR / "glaucoma_model.pkl"

CLASSES = ["Normal", "Glaucoma"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_dataset(split):
    features = []
    labels = []

    for label, class_name in enumerate(CLASSES):
        folder = DATASET_DIR / split / class_name

        if not folder.exists():
            raise FileNotFoundError(
                f"Dataset folder missing: {folder}"
            )

        files = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )

        print(f"{split}/{class_name}: {len(files)} images")

        for index, image_path in enumerate(files, start=1):
            try:
                img = preprocess_image(image_path)
                vector = extract_features(img)

                features.append(vector)
                labels.append(label)

            except Exception as exc:
                print(f"Skipping {image_path.name}: {exc}")

            if index % 200 == 0:
                print(f"  Processed {index}/{len(files)}")

    if not features:
        raise RuntimeError(f"No valid images loaded from {split}")

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)

    print(f"{split} feature matrix: {X.shape}")
    return X, y


def main():
    start = time.time()

    print("\nLoading training data...")
    X_train, y_train = load_dataset("train")

    print("\nLoading validation data...")
    X_val, y_val = load_dataset("val")

    print("\nTraining MLP model...")

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    val_predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, val_predictions)

    print("\n--- Validation Results ---")
    print(f"Validation accuracy: {accuracy * 100:.2f}%")
    print(
        classification_report(
            y_val,
            val_predictions,
            labels=[0, 1],
            target_names=CLASSES,
            zero_division=0
        )
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "classes": CLASSES,
            "feature_count": 128
        },
        MODEL_PATH
    )

    print(f"\nModel saved successfully:\n{MODEL_PATH}")
    print(f"Training iterations: {model.n_iter_}")
    print(f"Elapsed time: {(time.time() - start) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
