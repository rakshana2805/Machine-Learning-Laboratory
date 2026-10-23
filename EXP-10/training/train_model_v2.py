
from pathlib import Path
import sys
import warnings

import cv2
import joblib
import numpy as np

from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.ml.preprocessing import preprocess_image
from backend.ml.feature_extraction import extract_features


# Project paths
DATASET_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "backend" / "models"
MODEL_PATH = MODEL_DIR / "glaucoma_model.pkl"

# Must match the label order used by evaluation and prediction.
CLASSES = ["Glaucoma", "Normal"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
EXPECTED_FEATURES = 128
RANDOM_STATE = 42


def load_data(split):
    features = []
    labels = []

    print(f"\nLoading {split} dataset...")

    for label, class_name in enumerate(CLASSES):
        folder = DATASET_DIR / split / class_name

        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        image_files = sorted(
            path for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in EXTENSIONS
        )

        print(f"{split}/{class_name}: {len(image_files)} images")

        for image_path in image_files:
            try:
                image = preprocess_image(image_path)
                vector = extract_features(image)

                if vector.shape != (EXPECTED_FEATURES,):
                    raise ValueError(
                        f"Expected {EXPECTED_FEATURES} features, "
                        f"received {vector.shape}"
                    )

                if not np.all(np.isfinite(vector)):
                    raise ValueError("Feature vector contains invalid values")

                features.append(vector)
                labels.append(label)

            except Exception as error:
                print(f"Skipped {image_path.name}: {error}")

    if not features:
        raise RuntimeError(f"No usable images found in {split} dataset.")

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)

    print(f"{split} feature shape: {X.shape}")
    print(f"{split} labels shape: {y.shape}")

    return X, y


def build_model(layers, alpha):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=layers,
            activation="relu",
            solver="adam",
            alpha=alpha,
            learning_rate_init=0.0005,
            learning_rate="adaptive",
            max_iter=800,
            early_stopping=False,
            tol=0.0001,
            n_iter_no_change=30,
            random_state=RANDOM_STATE,
        )),
    ])


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_data("train")
    X_val, y_val = load_data("val")

    # Select the model using validation macro F1, not test accuracy.
    configurations = [
        {"layers": (32,), "alpha": 0.1},
        {"layers": (32, 16), "alpha": 0.1},
        {"layers": (64,), "alpha": 0.1},
        {"layers": (64, 32), "alpha": 0.01},
        {"layers": (64, 32), "alpha": 0.1},
        {"layers": (64, 32), "alpha": 1.0},
        {"layers": (128, 64), "alpha": 0.1},
        {"layers": (128, 64), "alpha": 1.0},
        {"layers": (128, 32), "alpha": 1.0},
        {"layers": (256, 128), "alpha": 0.01},
    ]

    best_model = None
    best_score = -1.0
    best_config = None

    print("\n========== MLP MODEL SELECTION ==========")

    for index, config in enumerate(configurations, start=1):
        print(
            f"\n[{index}/{len(configurations)}] "
            f"Layers={config['layers']}, alpha={config['alpha']}"
        )

        model = build_model(
            layers=config["layers"],
            alpha=config["alpha"],
        )

        try:
            model.fit(X_train, y_train)

            val_predictions = model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_predictions)
            val_f1 = f1_score(
                y_val,
                val_predictions,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )

            print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
            print(f"Validation macro F1: {val_f1:.4f}")
            print(
                "Training iterations:",
                model.named_steps["mlp"].n_iter_,
            )

            if val_f1 > best_score:
                best_score = val_f1
                best_model = model
                best_config = config.copy()

        except Exception as error:
            print(f"Configuration failed: {error}")

    if best_model is None:
        raise RuntimeError("All MLP configurations failed.")

    # Save the best model selected on validation data.
    train_predictions = best_model.predict(X_train)
    val_predictions = best_model.predict(X_val)

    saved_data = {
        "model": best_model,
        "classes": CLASSES,
        "feature_count": EXPECTED_FEATURES,
        "best_config": best_config,
        "validation_macro_f1": best_score,
    }

    joblib.dump(saved_data, MODEL_PATH)

    print("\n========== BEST MODEL ==========")
    print(f"Hidden layers: {best_config['layers']}")
    print(f"Alpha: {best_config['alpha']}")
    print(
        f"Training accuracy: "
        f"{accuracy_score(y_train, train_predictions) * 100:.2f}%"
    )
    print(
        f"Validation accuracy: "
        f"{accuracy_score(y_val, val_predictions) * 100:.2f}%"
    )
    print(f"Validation macro F1: {best_score:.4f}")

    print("\nValidation classification report:")
    print(classification_report(
        y_val,
        val_predictions,
        labels=[0, 1],
        target_names=CLASSES,
        digits=4,
        zero_division=0,
    ))

    print(f"Model saved successfully: {MODEL_PATH}")
    print("\nTraining completed.")


if __name__ == "__main__":
    main()
