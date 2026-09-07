from __future__ import annotations

import cv2
import numpy as np


def warp_image(image: np.ndarray, homography: np.ndarray, output_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = output_shape
    warped = cv2.warpPerspective(image, homography, (width, height))
    valid = cv2.warpPerspective(np.ones(image.shape[:2], dtype=np.uint8), homography, (width, height)) > 0
    return warped, valid

