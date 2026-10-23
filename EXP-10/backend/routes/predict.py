
import base64
import inspect
import json
import logging
import os
import tempfile
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
from flask import Blueprint, jsonify, request
from PIL import Image, ImageOps, UnidentifiedImageError

from backend.ml.preprocessing import preprocess_image
from backend.ml.feature_extraction import extract_features


logger = logging.getLogger(__name__)

predict_bp = Blueprint("predict", __name__)

PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_DIR / "backend" / "models" / "glaucoma_model.pkl"

EVALUATION_DIR = PROJECT_DIR / "backend" / "models" / "evaluation"
EVALUATION_FILE = EVALUATION_DIR / "evaluation_results.json"
HEATMAP_FILE = EVALUATION_DIR / "confusion_matrix_heatmap.png"
METRICS_CHART_FILE = EVALUATION_DIR / "metrics_bar_chart.png"

ALLOWED_EXTENSIONS = {
    "png", "jpg", "jpeg", "bmp", "tif", "tiff"
}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

_model_bundle = None


# =========================================================
# CUSTOM ERROR FOR INVALID UPLOADS
# =========================================================

class InvalidImageError(Exception):
    """Raised when uploaded image content is invalid."""


# =========================================================
# MODEL LOADING
# =========================================================

def load_model_bundle():
    """
    Load the existing saved model once.
    This function does not train or overwrite the model.
    """
    global _model_bundle

    if _model_bundle is not None:
        return _model_bundle

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Saved model not found: {MODEL_PATH}"
        )

    bundle = joblib.load(MODEL_PATH)

    if not isinstance(bundle, dict):
        raise ValueError(
            "The saved model file does not contain the expected dictionary bundle."
        )

    if "model" not in bundle:
        raise ValueError(
            "The saved model bundle does not contain the 'model' key."
        )

    model = bundle["model"]

    if not callable(getattr(model, "predict", None)):
        raise ValueError(
            "The saved model does not provide a callable predict() method."
        )

    _model_bundle = bundle

    logger.info("Saved model loaded from %s", MODEL_PATH)

    return _model_bundle


# =========================================================
# UPLOAD VALIDATION
# =========================================================

def allowed_file(filename):
    return (
        bool(filename)
        and "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def validate_image_bytes(image_bytes):
    """Verify that uploaded bytes contain a readable image."""

    if not image_bytes:
        raise InvalidImageError("The uploaded image is empty.")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()

        # Reopen after verify and force actual pixel decoding.
        with Image.open(BytesIO(image_bytes)) as image:
            image = ImageOps.exif_transpose(image)
            image.load()

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        raise InvalidImageError(
            "The uploaded file is not a valid readable image."
        ) from error


# =========================================================
# IMAGE TO DATA URL
# =========================================================

def image_to_data_url(image):
    """Convert an image or NumPy array to a PNG data URL."""

    if image is None:
        return None

    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")

    else:
        array = np.asarray(image)

        if array.size == 0:
            return None

        if not np.isfinite(array).all():
            array = np.nan_to_num(array)

        if array.dtype != np.uint8:
            if (
                np.issubdtype(array.dtype, np.floating)
                and array.min() >= 0
                and array.max() <= 1
            ):
                array = array * 255.0

            array = np.clip(array, 0, 255).astype(np.uint8)

        if array.ndim == 2:
            pil_image = Image.fromarray(array).convert("RGB")

        elif array.ndim == 3 and array.shape[2] == 4:
            pil_image = Image.fromarray(array).convert("RGB")

        elif array.ndim == 3 and array.shape[2] >= 3:
            pil_image = Image.fromarray(
                array[:, :, :3]
            ).convert("RGB")

        else:
            return None

    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    return f"data:image/png;base64,{encoded}"


# =========================================================
# PREPROCESSING AND STAGES
# =========================================================

def get_preprocessing_stages(image_path):
    """
    Support both preprocessing signatures:
      preprocess_image(path)
      preprocess_image(path, return_stages=True)

    The actual imported function signature determines how it is called.
    """

    try:
        signature = inspect.signature(preprocess_image)

        supports_stages = (
            "return_stages" in signature.parameters
            or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
        )

    except (TypeError, ValueError):
        # Safe fallback: do not guess that return_stages is accepted.
        supports_stages = False

    logger.info(
        "Preprocessing function: %s",
        inspect.getsourcefile(preprocess_image)
    )
    logger.info(
        "Preprocessing supports return_stages: %s",
        supports_stages
    )

    if supports_stages:
        result = preprocess_image(
            str(image_path),
            return_stages=True
        )
    else:
        result = preprocess_image(str(image_path))

    if isinstance(result, tuple):
        if len(result) < 2:
            raise RuntimeError(
                "Preprocessing returned an incomplete tuple."
            )

        processed_image = result[0]
        stages = result[1]

    elif isinstance(result, dict):
        stages = result.get("stages", {})
        processed_image = result.get("processed_image")

        if processed_image is None:
            processed_image = result.get("image")

    else:
        processed_image = result
        stages = {}

    if processed_image is None:
        raise RuntimeError(
            "Preprocessing did not return an image."
        )

    if not isinstance(stages, dict):
        stages = {}

    # Support common stage aliases.
    aliases = {
        "original": [
            "original_image", "upload"
        ],
        "rgb_conversion": [
            "rgb", "rgb_image"
        ],
        "resized": [
            "resize", "resized_image", "image_resizing"
        ],
        "gaussian_blur": [
            "blur", "blurred", "gaussian"
        ],
        "clahe_enhancement": [
            "clahe", "enhanced", "clahe_image"
        ],
        "normalized": [
            "normalization", "normalized_image"
        ],
    }

    normalized_stages = dict(stages)

    for canonical_name, alternatives in aliases.items():
        if normalized_stages.get(canonical_name) is None:
            for alternative in alternatives:
                if normalized_stages.get(alternative) is not None:
                    normalized_stages[canonical_name] = (
                        normalized_stages[alternative]
                    )
                    break

    # Always include the actual uploaded image in the preview.
    if normalized_stages.get("original") is None:
        with Image.open(image_path) as source:
            normalized_stages["original"] = (
                ImageOps.exif_transpose(source)
                .convert("RGB")
                .copy()
            )

    definitions = [
        (
            "original",
            "Original Image",
            "Uploaded retinal image."
        ),
        (
            "rgb_conversion",
            "RGB Conversion",
            "RGB color conversion stage."
        ),
        (
            "resized",
            "Image Resizing",
            "Resized image from the preprocessing pipeline."
        ),
        (
            "gaussian_blur",
            "Gaussian Blur",
            "Blur stage from the preprocessing pipeline."
        ),
        (
            "clahe_enhancement",
            "CLAHE Enhancement",
            "Contrast enhancement stage from preprocessing."
        ),
        (
            "normalized",
            "Normalization",
            "Normalized image returned by preprocessing."
        ),
    ]

    output_stages = []

    for stage_id, title, description in definitions:
        stage_image = normalized_stages.get(stage_id)

        output_stages.append({
            "id": stage_id,
            "name": title,
            "description": description,
            "status": (
                "completed"
                if stage_image is not None
                else "unavailable"
            ),
            "image": image_to_data_url(stage_image),
        })

    return processed_image, output_stages


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def to_feature_vector(processed_image, bundle=None):
    """
    Convert processed image to a validated (1, N) feature vector.
    """

    features = extract_features(processed_image)

    features = np.asarray(
        features,
        dtype=np.float64
    ).reshape(1, -1)

    if features.shape[1] == 0:
        raise RuntimeError(
            "Feature extraction returned no features."
        )

    if not np.isfinite(features).all():
        raise RuntimeError(
            "Feature extraction returned NaN or infinite values."
        )

    if bundle is None:
        bundle = load_model_bundle()

    model = bundle["model"]

    expected_count = bundle.get("feature_count")

    if expected_count is None:
        expected_count = getattr(
            model,
            "n_features_in_",
            None
        )

    if (
        expected_count is not None
        and features.shape[1] != int(expected_count)
    ):
        raise RuntimeError(
            "Feature count mismatch: the saved model expects "
            f"{int(expected_count)} features, but feature extraction "
            f"returned {features.shape[1]}. Check feature_extraction.py "
            "and the preprocessing used when training the model."
        )

    return features


# =========================================================
# CLASS LABELS
# =========================================================

def get_class_name(raw_prediction, class_names):
    """Map a model prediction to its display class label."""

    if isinstance(raw_prediction, np.generic):
        raw_prediction = raw_prediction.item()

    # Exact label match.
    for class_name in class_names:
        if str(class_name).casefold() == str(raw_prediction).casefold():
            return str(class_name)

    # Numeric class index -> display label.
    if isinstance(raw_prediction, (int, np.integer)):
        index = int(raw_prediction)

        if 0 <= index < len(class_names):
            return str(class_names[index])

    # Handle predictions such as "0" when they represent class indices.
    if isinstance(raw_prediction, str) and raw_prediction.isdigit():
        index = int(raw_prediction)

        if 0 <= index < len(class_names):
            return str(class_names[index])

    return str(raw_prediction)


# =========================================================
# CLASS PROBABILITIES
# =========================================================

def get_probabilities(model, features, class_names):
    """Return per-class probabilities when model supports predict_proba."""

    if not callable(getattr(model, "predict_proba", None)):
        return {}

    values = np.asarray(
        model.predict_proba(features)
    ).reshape(-1)

    model_classes = getattr(model, "classes_", class_names)

    probabilities = {}

    for index, value in enumerate(values):
        if index >= len(model_classes):
            continue

        label = get_class_name(
            model_classes[index],
            class_names
        )

        probabilities[label] = round(
            float(value) * 100.0,
            2
        )

    return probabilities


# =========================================================
# DATABASE HISTORY
# =========================================================

def save_history(filename, prediction, confidence, probabilities):
    """
    Save history using the project's existing DB helper.
    History failure must not hide a successful prediction.
    """

    try:
        from backend.db import save_prediction

        return save_prediction(
            filename=filename,
            prediction=prediction,
            confidence=confidence,
            probabilities=probabilities,
        )

    except ImportError:
        logger.warning(
            "backend.db.save_prediction is unavailable."
        )

    except Exception:
        logger.exception(
            "Prediction succeeded, but history could not be saved."
        )

    return None


# =========================================================
# POST /api/predict
# =========================================================

@predict_bp.route("/predict", methods=["POST"])
def predict():
    temp_path = None
    original_filename = None

    logger.info("Prediction request received.")

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "message": "No image received. Use the 'image' upload field.",
        }), 400

    uploaded_file = request.files["image"]

    if not uploaded_file.filename:
        return jsonify({
            "success": False,
            "message": "Please select an image.",
        }), 400

    original_filename = Path(
        uploaded_file.filename
    ).name

    if not allowed_file(original_filename):
        return jsonify({
            "success": False,
            "message": (
                "Unsupported image format. Use PNG, JPG, JPEG, BMP or TIFF."
            ),
        }), 400

    try:
        # 1. Read at most 10 MB plus one byte to detect oversized uploads.
        image_bytes = uploaded_file.read(
            MAX_UPLOAD_BYTES + 1
        )

        if not image_bytes:
            raise InvalidImageError(
                "The uploaded file is empty."
            )

        if len(image_bytes) > MAX_UPLOAD_BYTES:
            return jsonify({
                "success": False,
                "message": "Image exceeds the 10 MB upload limit.",
            }), 413

        # 2. Validate content instead of trusting the extension alone.
        validate_image_bytes(image_bytes)

        suffix = "." + original_filename.rsplit(
            ".", 1
        )[1].lower()

        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=suffix,
            delete=False
        ) as temporary_file:
            temporary_file.write(image_bytes)
            temp_path = temporary_file.name

        logger.info(
            "Image validated; starting preprocessing: %s",
            original_filename
        )

        # 3. Preprocess.
        processed_image, stages = get_preprocessing_stages(
            temp_path
        )

        # 4. Load existing model.
        bundle = load_model_bundle()
        model = bundle["model"]

        class_names = bundle.get("classes")

        if not class_names:
            class_names = list(
                getattr(model, "classes_", [])
            )

        if not class_names:
            raise RuntimeError(
                "The saved model bundle does not contain class labels."
            )

        # 5. Extract model features.
        features = to_feature_vector(
            processed_image,
            bundle=bundle
        )

        logger.info(
            "Features extracted: shape=%s",
            features.shape
        )

        # 6. Predict class.
        raw_prediction = model.predict(features)[0]

        prediction_name = get_class_name(
            raw_prediction,
            class_names
        )

        # 7. Predict probabilities if supported.
        probabilities = get_probabilities(
            model,
            features,
            class_names
        )

        confidence = probabilities.get(
            prediction_name
        )

        if confidence is None and probabilities:
            confidence = max(
                probabilities.values()
            )

        # 8. Format feature details for the dashboard.
        feature_values = [
            round(float(value), 6)
            for value in features.reshape(-1)
        ]

        group_size = 16
        feature_groups = []

        for start in range(
            0,
            len(feature_values),
            group_size
        ):
            values = feature_values[
                start:start + group_size
            ]

            feature_groups.append({
                "name": (
                    f"Features {start + 1}-"
                    f"{start + len(values)}"
                ),
                "start": start + 1,
                "count": len(values),
                "values": values,
            })

        # 9. Save history. A DB issue should not cancel the prediction.
        history_id = save_history(
            filename=original_filename,
            prediction=prediction_name,
            confidence=confidence,
            probabilities=probabilities,
        )

        logger.info(
            "Prediction completed: filename=%s, prediction=%s, "
            "features=%d",
            original_filename,
            prediction_name,
            len(feature_values)
        )

        return jsonify({
            "success": True,
            "filename": original_filename,
            "prediction": prediction_name,
            "raw_prediction": str(raw_prediction),
            "confidence": confidence,
            "classes": [
                str(label) for label in class_names
            ],
            "probabilities": probabilities,
            "feature_count": len(feature_values),
            "features": feature_values,
            "feature_values": feature_values,
            "feature_groups": feature_groups,
            "stages": stages,
            "history_id": history_id,
            "message": "Prediction completed successfully.",
        }), 200

    except InvalidImageError as error:
        logger.warning(
            "Invalid upload: %s",
            error
        )

        return jsonify({
            "success": False,
            "message": str(error),
        }), 400

    except Exception:
        # The complete traceback will be printed in the Flask terminal.
        logger.exception(
            "Unexpected error during prediction for %s",
            original_filename
        )

        return jsonify({
            "success": False,
            "message": (
                "Prediction failed because of a server-side error. "
                "Check the Flask terminal for the complete traceback."
            ),
        }), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)

            except OSError:
                logger.warning(
                    "Could not remove temporary image: %s",
                    temp_path
                )


# =========================================================
# GET /api/history
# =========================================================

@predict_bp.route("/history", methods=["GET"])
def prediction_history():
    try:
        from backend.db import get_prediction_history

        limit = request.args.get(
            "limit",
            default=100,
            type=int
        )

        if limit is None:
            limit = 100

        limit = max(1, min(limit, 500))

        records = get_prediction_history(
            limit=limit
        )

        return jsonify({
            "success": True,
            "count": len(records),
            "storage": "SQLite",
            "history": records,
        }), 200

    except Exception:
        logger.exception(
            "Could not load prediction history."
        )

        return jsonify({
            "success": False,
            "message": "Could not load prediction history.",
            "history": [],
        }), 500


# =========================================================
# EVALUATION HELPERS
# =========================================================

def image_as_data_url(path):
    if not path.is_file():
        return None

    encoded = base64.b64encode(
        path.read_bytes()
    ).decode("ascii")

    return f"data:image/png;base64,{encoded}"


# =========================================================
# GET /api/evaluation
# =========================================================

@predict_bp.route("/evaluation", methods=["GET"])
def get_evaluation_results():
    """Return saved evaluation results without retraining the model."""

    if not EVALUATION_FILE.is_file():
        return jsonify({
            "success": False,
            "message": "Saved evaluation results were not found.",
        }), 404

    try:
        with EVALUATION_FILE.open(
            "r",
            encoding="utf-8"
        ) as file:
            results = json.load(file)

        if not isinstance(results, dict):
            raise ValueError(
                "Evaluation JSON must contain a JSON object."
            )

        results["success"] = True

        results["confusion_matrix_image"] = image_as_data_url(
            HEATMAP_FILE
        )

        results["metrics_chart_image"] = image_as_data_url(
            METRICS_CHART_FILE
        )

        return jsonify(results), 200

    except Exception:
        logger.exception(
            "Could not read saved evaluation results."
        )

        return jsonify({
            "success": False,
            "message": "Could not read the saved evaluation results.",
        }), 500
