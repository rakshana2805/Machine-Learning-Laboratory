from pathlib import Path
import hashlib

import joblib
import numpy as np

from backend.ml.preprocessing import preprocess_image
from backend.ml.feature_extraction import extract_features


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "backend" / "models" / "glaucoma_model.pkl"
NORMAL_DIR = ROOT / "dataset" / "test" / "normal"


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    images = sorted(
        p for p in NORMAL_DIR.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
        }
    )

    if not images:
        raise FileNotFoundError(
            f"No test images found in {NORMAL_DIR}"
        )

    image_path = images[0]
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    classes = bundle.get("classes", ["Glaucoma", "Normal"])

    print("Image:", image_path)
    print("Model:", MODEL_PATH)
    print("Classes:", classes)
    print("-" * 50)

    previous_hash = None
    previous_prediction = None

    for attempt in range(1, 6):
        processed = preprocess_image(str(image_path))

        if isinstance(processed, tuple):
            processed = processed[0]

        features = np.asarray(
            extract_features(processed),
            dtype=np.float64
        ).reshape(1, -1)

        feature_hash = hashlib.sha256(
            features.tobytes()
        ).hexdigest()[:16]

        raw_prediction = model.predict(features)[0]

        if isinstance(raw_prediction, np.generic):
            raw_prediction = raw_prediction.item()

        if str(raw_prediction) in {"0", "1"}:
            prediction = classes[int(raw_prediction)]
        else:
            prediction = str(raw_prediction)

        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            model_classes = getattr(model, "classes_", range(len(probabilities)))

            probability = {
                str(classes[int(label)]) if str(label) in {"0", "1"}
                else str(label): round(float(prob) * 100, 2)
                for label, prob in zip(model_classes, probabilities)
            }

        print(f"Run {attempt}")
        print("Feature hash:", feature_hash)
        print("Raw prediction:", raw_prediction)
        print("Prediction:", prediction)
        print("Probabilities:", probability)

        if previous_hash is not None:
            print("Features unchanged:", feature_hash == previous_hash)
            print(
                "Prediction unchanged:",
                prediction == previous_prediction
            )

        previous_hash = feature_hash
        previous_prediction = prediction
        print("-" * 50)


if __name__ == "__main__":
    main()