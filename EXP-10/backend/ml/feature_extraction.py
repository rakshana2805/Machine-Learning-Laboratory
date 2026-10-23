
import cv2
import numpy as np


def extract_features(img):
    # Input: normalized RGB image, shape (224, 224, 3)
    if img.shape != (224, 224, 3):
        raise ValueError(
            "Expected RGB image with shape (224, 224, 3)"
        )

    rgb = np.clip(img * 255, 0, 255).astype(np.uint8)

    # 1. Color histograms: 32 bins x 3 channels = 96 features
    color_features = []

    for channel in range(3):
        hist = cv2.calcHist(
            [img.astype(np.float32)],
            [channel],
            None,
            [32],
            [0, 1]
        ).flatten()

        hist = hist / (hist.sum() + 1e-8)
        color_features.extend(hist)

    # Convert to grayscale
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Sobel gradients
    gx = cv2.Sobel(
        gray, cv2.CV_32F, 1, 0, ksize=3
    )
    gy = cv2.Sobel(
        gray, cv2.CV_32F, 0, 1, ksize=3
    )

    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    # 2. Gradient magnitude histogram: 14 features
    mag_hist, _ = np.histogram(
        magnitude,
        bins=14,
        range=(0, 1443)
    )
    mag_hist = mag_hist.astype(np.float32)
    mag_hist /= mag_hist.sum() + 1e-8

    # 3. Gradient orientation histogram: 16 features
    angle = (
        np.degrees(np.arctan2(gy, gx)) + 360
    ) % 360

    angle_hist, _ = np.histogram(
        angle,
        bins=16,
        range=(0, 360)
    )
    angle_hist = angle_hist.astype(np.float32)
    angle_hist /= angle_hist.sum() + 1e-8

    # 4. Edge energy statistics: 2 features
    edge_mean = float(np.mean(magnitude)) / 1443.0
    edge_std = float(np.std(magnitude)) / 1443.0

    features = np.concatenate([
        np.asarray(color_features, dtype=np.float32),
        mag_hist,
        angle_hist,
        np.array([edge_mean, edge_std], dtype=np.float32)
    ])

    assert features.shape == (128,)
    return features