from __future__ import annotations

import cv2
import numpy as np


def compute_feature_change_map(features_before: np.ndarray, features_after: np.ndarray,
                               output_shape: tuple[int, int]) -> np.ndarray:
    if features_before.shape != features_after.shape:
        raise ValueError("Feature maps must have identical shapes")
    similarity = np.sum(features_before * features_after, axis=-1)
    difference = np.clip(1.0 - similarity, 0.0, 1.0)
    return cv2.resize(difference.astype(np.float32), (output_shape[1], output_shape[0]), interpolation=cv2.INTER_CUBIC)

