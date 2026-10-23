
from pathlib import Path

import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from backend.ml.preprocessing import preprocess_image
from backend.ml.feature_extraction import extract_features


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "backend" / "models"

# Preserve the existing model.
CANDIDATE_MODEL_PATH = MODEL_DIR / "glaucoma_model_candidate.pkl"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

# Keep labels consistent with the existing project.
CLASS_TO_ID = {
    "glaucoma": 0,
    "normal": 1,
}

CLASS_NAMES = ["Glaucoma", "Normal"]
EXPECTED_FEATURES = 128


def preprocess_for_features(image_path):
    """Use the same preprocessing and feature extractor as prediction."""

    result = preprocess_image(str(image_path))

    # Accommodate preprocessing implementations that return stages.
    if isinstance(result, tuple):
        result = result[0]

    if result is None:
        raise ValueError("Preprocessing returned no image.")

    return result


def load_split(split_dir):
    """Load features and labels from split/class_name/image files."""

    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset split not found: {split_dir}"
        )

    features = []
    labels = []
    errors = []

    class_directories = {
        folder.name.casefold(): folder
        for folder in split_dir.iterdir()
        if folder.is_dir()
    }

    for class_name, class_id in CLASS_TO_ID.items():
        folder = class_directories.get(class_name)

        if folder is None:
            raise FileNotFoundError(
                f"Missing class folder '{class_name}' in {split_dir}"
            )

        image_paths = sorted(
            path for path in folder.rglob("*")
            if path.is_file()
            and path.suffix.casefold() in IMAGE_EXTENSIONS
        )

        print(
            f"{split_dir.name}/{class_name}: "
            f"{len(image_paths)} images"
        )

        for image_path in image_paths:
            try:
                image = preprocess_for_features(image_path)
                feature_vector = np.asarray(
                    extract_features(image),
                    dtype=np.float32
                ).reshape(-1)

                if feature_vector.shape != (EXPECTED_FEATURES,):
                    raise ValueError(
                        f"Expected {EXPECTED_FEATURES} features, "
                        f"got {feature_vector.shape}"
                    )

                if not np.isfinite(feature_vector).all():
                    raise ValueError(
                        "Features contain NaN or infinite values."
                    )

                features.append(feature_vector)
                labels.append(class_id)

            except Exception as error:
                errors.append((image_path, str(error)))

    if errors:
        print(f"\nSkipped {len(errors)} unreadable/invalid images:")
        for path, error in errors[:10]:
            print(f"  {path}: {error}")

    if not features:
        raise RuntimeError(
            f"No usable images were found in {split_dir}"
        )

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)

    if len(np.unique(y)) != 2:
        raise RuntimeError(
            f"Both Glaucoma and Normal classes are required in {split_dir}"
        )

    print(f"Loaded {len(y)} images; feature shape: {X.shape}")

    return X, y


def evaluate(model, X, y, title):
    """Print validation metrics and a confusion matrix."""

    predictions = model.predict(X)

    print(f"\n========== {title} ==========")
    print(f"Accuracy: {accuracy_score(y, predictions) * 100:.2f}%")

    print("\nClassification report:")
    print(
        classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix (rows=actual, columns=predicted):")
    print("                 Predicted Glaucoma   Predicted Normal")

    matrix = confusion_matrix(
        y,
        predictions,
        labels=[0, 1]
    )

    print(f"Actual Glaucoma       {matrix[0, 0]:6d}             {matrix[0, 1]:6d}")
    print(f"Actual Normal         {matrix[1, 0]:6d}             {matrix[1, 1]:6d}")

    return matrix


def find_validation_dir():
    for folder_name in ("val", "validation", "valid"):
        folder = DATASET_DIR / folder_name
        if folder.is_dir():
            return folder

    return None


def main():
    print("=" * 60)
    print("GLAUCOMA CLASSIFIER - CANDIDATE MODEL TRAINING")
    print("=" * 60)

    # Load the training split only.
    X_train, y_train = load_split(DATASET_DIR / "train")

    validation_dir = find_validation_dir()

    if validation_dir:
        # Use a separate validation split when available.
        X_val, y_val = load_split(validation_dir)

    else:
        # Otherwise reserve a stratified part of train data for validation.
        print(
            "\nNo val/validation/valid folder found. "
            "Using a stratified training holdout."
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.20,
            random_state=42,
            stratify=y_train,
        )

    print("\nTraining features:", X_train.shape)
    print("Validation features:", X_val.shape)

    # Scaling and classification are fitted together in one pipeline.
    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            SVC(
                kernel="rbf",
                C=1.0,
                class_weight="balanced",
                probability=True,
                random_state=42,
            ),
        ),
    ])

    print("\nTraining model...")
    model.fit(X_train, y_train)

    # Validation must happen before replacing the existing model.
    evaluate(
        model,
        X_train,
        y_train,
        "TRAINING METRICS"
    )

    evaluate(
        model,
        X_val,
        y_val,
        "VALIDATION METRICS"
    )

    bundle = {
        "model": model,
        "classes": CLASS_NAMES,
        "feature_count": EXPECTED_FEATURES,
        "feature_extractor": "spatial_gray_green_histogram_128",
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save as a candidate; do not overwrite glaucoma_model.pkl.
    joblib.dump(bundle, CANDIDATE_MODEL_PATH)

    print("\nCandidate model saved to:")
    print(CANDIDATE_MODEL_PATH)

    print("\nExisting glaucoma_model.pkl was not changed.")
    print(
        "Next: evaluate the candidate on the held-out test set "
        "before using it in Flask."
    )


if __name__ == "__main__":
    main()
