from pathlib import Path
import inspect
import joblib
import numpy as np

from backend.ml.preprocessing import preprocess_image
from backend.ml.feature_extraction import extract_features

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "backend" / "models" / "glaucoma_model.pkl"
TEST_DIR = ROOT / "dataset" / "test"

LABELS = ["Glaucoma", "Normal"]


def get_bundle():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    bundle = joblib.load(MODEL_PATH)

    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("Unexpected model bundle format.")

    return bundle


def preprocess(path):
    params = inspect.signature(preprocess_image).parameters

    if "return_stages" in params:
        result = preprocess_image(str(path), return_stages=True)
    else:
        result = preprocess_image(str(path))

    # The prediction pipeline expects the image, not its stages.
    if isinstance(result, tuple):
        result = result[0]

    if result is None:
        raise ValueError("Preprocessing returned no image.")

    return result


def map_prediction(raw, class_names):
    if isinstance(raw, np.generic):
        raw = raw.item()

    # Preserve labels that the model already predicts.
    for name in class_names:
        if str(name).casefold() == str(raw).casefold():
            return str(name)

    # Map numeric output only when the saved class order is known.
    if isinstance(raw, (int, np.integer)):
        index = int(raw)
        if 0 <= index < len(class_names):
            return str(class_names[index])

    return str(raw)


def main():
    bundle = get_bundle()
    model = bundle["model"]

    bundle_classes = bundle.get("classes")
    model_classes = getattr(model, "classes_", None)

    print("=" * 55)
    print("GLAUCOMA PREDICTION DIAGNOSTIC")
    print("=" * 55)
    print("Model:", MODEL_PATH)
    print("Preprocessing file:", inspect.getfile(preprocess_image))
    print("Preprocessing signature:", inspect.signature(preprocess_image))
    print("Bundle classes:", bundle_classes)
    print("Model classes_:", model_classes)
    print("Expected features:", getattr(model, "n_features_in_", None))

    # Prefer the bundle's class names; otherwise use model labels.
    class_names = bundle_classes
    if not class_names and model_classes is not None:
        class_names = list(model_classes)

    if not class_names:
        raise ValueError("Cannot determine saved model class labels.")

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    correct = 0
    total = 0
    counts = {}
    errors = []

    for folder_name, true_label in [
        ("glaucoma", "Glaucoma"),
        ("normal", "Normal"),
    ]:
        folder = TEST_DIR / folder_name

        if not folder.is_dir():
            print(f"Missing test folder: {folder}")
            continue

        files = sorted(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in valid_extensions
        )

        print(f"\nTesting {true_label}: {len(files)} images")

        for path in files:
            try:
                image = preprocess(path)
                features = np.asarray(
                    extract_features(image), dtype=np.float64
                ).reshape(1, -1)

                if features.shape[1] == 0:
                    raise ValueError("No features extracted.")

                if not np.isfinite(features).all():
                    raise ValueError("Non-finite feature values.")

                expected = bundle.get("feature_count")
                if expected is None:
                    expected = getattr(model, "n_features_in_", None)

                if expected is not None and features.shape[1] != int(expected):
                    raise ValueError(
                        f"Feature mismatch: expected {expected}, "
                        f"got {features.shape[1]}"
                    )

                raw = model.predict(features)[0]
                predicted = map_prediction(raw, class_names)

                total += 1
                counts[(true_label, predicted)] = (
                    counts.get((true_label, predicted), 0) + 1
                )

                if predicted.casefold() == true_label.casefold():
                    correct += 1

                if total <= 12:
                    print(
                        f"{path.name}: actual={true_label}, "
                        f"raw={raw}, predicted={predicted}, "
                        f"feature_count={features.shape[1]}"
                    )

            except Exception as error:
                errors.append((str(path), str(error)))

    print("\n" + "=" * 55)
    print("CONFUSION COUNTS (Actual -> Predicted)")
    for actual in LABELS:
        for predicted in LABELS:
            print(
                f"{actual:10} -> {predicted:10}: "
                f"{counts.get((actual, predicted), 0)}"
            )

    if total:
        print(f"\nDiagnostic accuracy: {correct / total * 100:.2f}%")
        print(f"Images tested successfully: {total}")

    print(f"Image errors: {len(errors)}")
    for path, error in errors[:10]:
        print(f"ERROR: {path} | {error}")

    print("\nNo model or dataset files were changed.")


if __name__ == "__main__":
    main()