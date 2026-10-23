
import cv2
import numpy as np
from pathlib import Path


def preprocess_image(
    image_path,
    target_size=(224, 224),
    return_stages=False
):
    """
    Preprocess a retinal fundus image.

    Default return:
        RGB float32 image with shape (height, width, 3)
        and pixel values in the [0, 1] range.

    If return_stages=True:
        Returns (normalized_image, stages).
    """

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    if (
        not isinstance(target_size, (tuple, list))
        or len(target_size) != 2
    ):
        raise ValueError(
            "target_size must be (width, height)."
        )

    width, height = map(int, target_size)

    if width <= 0 or height <= 0:
        raise ValueError(
            "Image width and height must be positive."
        )

    # Read image
    original_bgr = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR
    )

    if original_bgr is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    # Convert BGR to RGB
    original_rgb = cv2.cvtColor(
        original_bgr,
        cv2.COLOR_BGR2RGB
    )

    # Resize image
    resized_image = cv2.resize(
        original_rgb,
        (width, height),
        interpolation=cv2.INTER_AREA
    )

    # Gaussian blur
    blurred_image = cv2.GaussianBlur(
        resized_image,
        (5, 5),
        0
    )

    # Convert RGB to LAB
    lab_image = cv2.cvtColor(
        blurred_image,
        cv2.COLOR_RGB2LAB
    )

    l_channel, a_channel, b_channel = cv2.split(
        lab_image
    )

    # CLAHE enhancement
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge(
        (enhanced_l, a_channel, b_channel)
    )

    enhanced_rgb = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2RGB
    )

    # Correct format for feature_extraction.py:
    # RGB float32, normalized to [0, 1].
    normalized_image = np.ascontiguousarray(
        enhanced_rgb.astype(np.float32) / 255.0
    )

    if not return_stages:
        return normalized_image

    stages = {
        "original": original_rgb,
        "rgb_conversion": original_rgb,
        "resized": resized_image,
        "gaussian_blur": blurred_image,
        "clahe_enhancement": enhanced_rgb,
        "normalized": normalized_image,
    }

    return normalized_image, stages
