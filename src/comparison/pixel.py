from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def compute_pixel_maps(before: np.ndarray, after: np.ndarray) -> dict[str, np.ndarray]:
    before = cv2.resize(before, (after.shape[1], after.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    after_float = after.astype(np.float32) / 255.0
    rgb = np.mean(np.abs(before - after_float), axis=2)
    gray_before = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
    gray_after = cv2.cvtColor(after_float, cv2.COLOR_RGB2GRAY)
    grayscale = np.abs(gray_before - gray_after)
    try:
        _, _, ssim_map = structural_similarity(gray_before, gray_after, data_range=1.0, full=True)
        ssim = 1.0 - ssim_map.astype(np.float32)
    except ValueError:
        ssim = grayscale
    return {"rgb": np.clip(rgb, 0, 1), "grayscale": np.clip(grayscale, 0, 1), "ssim": np.clip(ssim, 0, 1)}

