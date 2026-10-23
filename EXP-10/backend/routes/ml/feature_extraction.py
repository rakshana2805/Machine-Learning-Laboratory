
from pathlib import Path

import cv2
import numpy as np


IMAGE_SIZE = (224, 224)
FEATURE_COUNT = 128
GRID_SIZE = 4
BINS_PER_CHANNEL = 4


def _load_rgb_image(image):
    """
    Accept an image path, NumPy array, or PIL image.
    Return an RGB uint8 image.
    """

    if isinstance(image, (str, Path)):
        image_path = Path(image)

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        bgr = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR
        )

        if bgr is None:
            raise ValueError(
                f"Unable to read image: {image_path}"
            )

        rgb = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2RGB
        )

    else:
        # Supports NumPy arrays and PIL images.
        if hasattr(image, "convert") and hasattr(image, "size"):
            image = image.convert("RGB")
            rgb = np.asarray(image)
        else:
            rgb = np.asarray(image)

        if rgb.size == 0:
            raise ValueError("Input image is empty.")

        if rgb.ndim == 2:
            rgb = cv2.cvtColor(
                rgb.astype(np.uint8),
                cv2.COLOR_GRAY2RGB
            )

        elif rgb.ndim == 3 and rgb.shape[2] == 4:
            rgb = cv2.cvtColor(
                rgb.astype(np.uint8),
                cv2.COLOR_RGBA2RGB
            )

        elif rgb.ndim == 3 and rgb.shape[2] == 3:
            pass

        else:
            raise ValueError(
                "Expected a grayscale, RGB, or RGBA image."
            )

        if np.issubdtype(rgb.dtype, np.floating):
            if not np.isfinite(rgb).all():
                raise ValueError(
                    "Image contains NaN or infinite values."
                )

            if rgb.min() >= 0 and rgb.max() <= 1:
                rgb = rgb * 255.0

        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(rgb)


def extract_features(image):
    """
    Extract exactly 128 handcrafted spatial histogram features.

    The 224x224 RGB image is divided into 4x4 cells.
    Each cell contributes:
      - 4 grayscale histogram bins
      - 4 green-channel histogram bins

    Total: 16 cells * 8 features = 128.

    IMPORTANT:
    This defines a new feature representation. The saved classifier
    must be trained using this exact extraction method before these
    features are used for prediction.
    """

    rgb = _load_rgb_image(image)

    rgb = cv2.resize(
        rgb,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY
    )

    green = rgb[:, :, 1]

    gray_cells = np.array_split(
        gray, GRID_SIZE, axis=0
    )
    green_cells = np.array_split(
        green, GRID_SIZE, axis=0
    )

    features = []

    for row in range(GRID_SIZE):
        gray_row = np.array_split(
            gray_cells[row], GRID_SIZE, axis=1
        )
        green_row = np.array_split(
            green_cells[row], GRID_SIZE, axis=1
        )

        for col in range(GRID_SIZE):
            for channel_cell in (
                gray_row[col],
                green_row[col]
            ):
                histogram = cv2.calcHist(
                    [channel_cell],
                    [0],
                    None,
                    [BINS_PER_CHANNEL],
                    [0, 256]
                ).flatten()

                total = float(histogram.sum())

                if total > 0:
                    histogram = histogram / total

                features.extend(
                    histogram.astype(np.float32).tolist()
                )

    feature_vector = np.asarray(
        features,
        dtype=np.float32
    )

    if feature_vector.shape != (FEATURE_COUNT,):
        raise RuntimeError(
            f"Expected {FEATURE_COUNT} features, "
            f"received shape {feature_vector.shape}."
        )

    if not np.isfinite(feature_vector).all():
        raise ValueError(
            "Feature vector contains invalid numerical values."
        )

    return feature_vector
