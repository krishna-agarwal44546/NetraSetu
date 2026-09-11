import cv2
import numpy as np


def calculate_quality(image_path):
    """
    Simple prototype quality assessment.

    Returns:
        usable
        quality
        sharpness_score
        brightness
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Could not read image")

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # -------------------------
    # Sharpness
    # -------------------------

    sharpness = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    # -------------------------
    # Brightness
    # -------------------------

    brightness = float(
        np.mean(gray)
    )

    # -------------------------
    # Prototype thresholds
    # -------------------------

    sharpness_ok = sharpness >= 50

    brightness_ok = (
        30 <= brightness <= 230
    )

    usable = (
        sharpness_ok and
        brightness_ok
    )

    if usable:
        quality = "Good"
    else:
        quality = "Poor"

    return {
        "usable": bool(usable),
        "quality": quality,
        "sharpnessScore": round(
            float(sharpness),
            2
        ),
        "brightness": round(
            brightness,
            2
        )
    }