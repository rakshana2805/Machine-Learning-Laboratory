
import json
import inspect
from pathlib import Path

import cv2
import joblib
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from backend.ml.preprocessing import preprocess_image
from backend.ml.feature_extraction import extract_features


# --------------------------------------------------
# Paths and saved model
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_DIR / "dataset"
TEST_DIR = DATASET_DIR / "test"

MODEL_PATH = (
    PROJECT_DIR / "backend" / "models" / "glaucoma_model.pkl"
)

OUTPUT_DIR = PROJECT_DIR / "backend" / "models" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_JSON = OUTPUT_DIR / "evaluation_results.json"
CONFUSION_PNG = OUTPUT_DIR / "confusion_matrix_heatmap.png"
METRICS_PNG = OUTPUT_DIR / "metrics_bar_chart.png"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}


# --------------------------------------------------
# Load model without retraining
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Saved model not found: {MODEL_PATH}"
    )

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test dataset not found: {TEST_DIR}"
    )

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
EXPECTED_FEATURES = int(bundle["feature_count"])
CLASSES = [str(c) for c in bundle["classes"]]

if len(CLASSES) != 2:
    raise ValueError(
        f"Expected two classes, found: {CLASSES}"
    )


def resolve_label(label):
    """Map a predicted label to the saved class name."""

    label_string = str(label)

    for class_name in CLASSES:
        if label_string.casefold() == class_name.casefold():
            return class_name

    try:
        index = int(label)
        if 0 <= index < len(CLASSES):
            return CLASSES[index]
    except (ValueError, TypeError, OverflowError):
        pass

    raise ValueError(
        f"Could not map model label {label!r} to {CLASSES}"
    )


def get_image_files(folder):
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def extract_model_features(image_path):
    """Use the same preprocessing and feature extractor as prediction."""

    parameters = inspect.signature(
        preprocess_image
    ).parameters

    if "return_stages" in parameters:
        result = preprocess_image(
            str(image_path),
            return_stages=False
        )
    else:
        result = preprocess_image(str(image_path))

    # Some implementations may return (image, stages).
    if isinstance(result, tuple):
        processed_image = result[0]
    else:
        processed_image = result

    features = np.asarray(
        extract_features(processed_image),
        dtype=np.float32
    ).reshape(-1)

    if features.size != EXPECTED_FEATURES:
        raise ValueError(
            f"{image_path.name}: expected {EXPECTED_FEATURES} "
            f"features, got {features.size}"
        )

    if not np.all(np.isfinite(features)):
        raise ValueError(
            f"{image_path.name}: features contain invalid values"
        )

    return features


# --------------------------------------------------
# Evaluate the held-out test dataset
# --------------------------------------------------

y_true = []
y_pred = []
processed_count = 0
skipped_files = []

print(f"Model: {MODEL_PATH}")
print(f"Test dataset: {TEST_DIR}")
print(f"Saved classes: {CLASSES}")
print("\nProcessing test images...\n")

class_folders = [
    folder for folder in TEST_DIR.iterdir()
    if folder.is_dir()
]

if not class_folders:
    raise ValueError(
        f"No class folders found inside {TEST_DIR}"
    )

for class_folder in sorted(class_folders):
    actual_folder_label = class_folder.name

    # Match the folder name to the model's saved class name.
    matching_classes = [
        class_name for class_name in CLASSES
        if class_name.casefold()
        == actual_folder_label.casefold()
    ]

    if not matching_classes:
        print(
            f"Skipping unknown class folder: {class_folder.name}"
        )
        continue

    actual_label = matching_classes[0]
    image_files = get_image_files(class_folder)

    print(
        f"{class_folder.name}: "
        f"{len(image_files)} image(s)"
    )

    for image_path in image_files:
        try:
            features = extract_model_features(image_path)
            prediction = model.predict(
                features.reshape(1, -1)
            )[0]

            predicted_label = resolve_label(prediction)

            y_true.append(actual_label)
            y_pred.append(predicted_label)
            processed_count += 1

        except Exception as error:
            skipped_files.append({
                "filename": str(image_path),
                "error": str(error)
            })
            print(f"Skipped {image_path.name}: {error}")


if processed_count == 0:
    raise RuntimeError(
        "No test images were successfully evaluated. "
        "Check the dataset paths and preprocessing code."
    )

if skipped_files:
    print(
        f"\nWarning: {len(skipped_files)} image(s) "
        "could not be evaluated."
    )


# --------------------------------------------------
# Calculate real evaluation metrics
# --------------------------------------------------

# Keep the same class order as the saved model.
labels = CLASSES

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels
)

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    labels=labels,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    labels=labels,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    labels=labels,
    average="macro",
    zero_division=0
)

report = classification_report(
    y_true,
    y_pred,
    labels=labels,
    output_dict=True,
    zero_division=0
)

print("\n========== TEST EVALUATION ==========")
print(f"Images evaluated : {processed_count}")
print(f"Images skipped   : {len(skipped_files)}")
print(f"Accuracy         : {accuracy * 100:.2f}%")
print(f"Macro Precision  : {precision * 100:.2f}%")
print(f"Macro Recall     : {recall * 100:.2f}%")
print(f"Macro F1-score   : {f1 * 100:.2f}%")
print("\nConfusion matrix:")
print(cm)

print("\nPer-class report:")
print(classification_report(
    y_true,
    y_pred,
    labels=labels,
    zero_division=0
))


# --------------------------------------------------
# Confusion matrix heatmap
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 6))

image = ax.imshow(cm, interpolation="nearest")

ax.set_title("Confusion Matrix — Test Dataset")
fig.colorbar(image, ax=ax, label="Number of Images")

ax.set(
    xticks=np.arange(len(labels)),
    yticks=np.arange(len(labels)),
    xticklabels=labels,
    yticklabels=labels,
    xlabel="Predicted Label",
    ylabel="Actual Label"
)

threshold = cm.max() / 2 if cm.size else 0

for row in range(cm.shape[0]):
    for col in range(cm.shape[1]):
        ax.text(
            col,
            row,
            str(cm[row, col]),
            ha="center",
            va="center",
            color=(
                "white" if cm[row, col] > threshold
                else "black"
            ),
            fontsize=13
        )

plt.tight_layout()
plt.savefig(CONFUSION_PNG, dpi=160, bbox_inches="tight")
plt.close(fig)


# --------------------------------------------------
# Metrics comparison bar chart
# --------------------------------------------------

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score"
]

metric_values = [
    accuracy * 100,
    precision * 100,
    recall * 100,
    f1 * 100
]

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(metric_names, metric_values)

ax.set_title("Model Evaluation Metrics")
ax.set_ylabel("Score (%)")
ax.set_ylim(0, 100)
ax.grid(axis="y", linestyle="--", alpha=0.3)

for bar, value in zip(bars, metric_values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{value:.2f}%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(METRICS_PNG, dpi=160, bbox_inches="tight")
plt.close(fig)


# --------------------------------------------------
# Save results for the Flask dashboard
# --------------------------------------------------

results = {
    "success": True,
    "dataset": str(TEST_DIR),
    "model_path": str(MODEL_PATH),
    "classes": labels,
    "images_evaluated": processed_count,
    "images_skipped": len(skipped_files),
    "metrics": {
        "accuracy": round(float(accuracy * 100), 2),
        "precision": round(float(precision * 100), 2),
        "recall": round(float(recall * 100), 2),
        "f1_score": round(float(f1 * 100), 2)
    },
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "artifacts": {
        "confusion_matrix_heatmap": str(CONFUSION_PNG),
        "metrics_bar_chart": str(METRICS_PNG)
    },
    "skipped_files": skipped_files
}

with open(RESULT_JSON, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("\n========== OUTPUT FILES ==========")
print(f"Results JSON : {RESULT_JSON}")
print(f"Confusion map: {CONFUSION_PNG}")
print(f"Metrics chart: {METRICS_PNG}")
print("\nEvaluation completed.")
print("The saved model was not retrained or overwritten.")
