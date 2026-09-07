from __future__ import annotations

import numpy as np


def compute_fused_change_map(dino: np.ndarray, ssim: np.ndarray, rgb: np.ndarray,
                             dino_weight: float = 0.6, ssim_weight: float = 0.3,
                             rgb_weight: float = 0.1) -> np.ndarray:
    weights = np.array([dino_weight, ssim_weight, rgb_weight], dtype=np.float32)
    if np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("Change-map weights must be non-negative and have a positive sum")
    weights /= weights.sum()
    maps = [np.nan_to_num(np.asarray(item, dtype=np.float32), nan=0.0) for item in (dino, ssim, rgb)]
    return np.clip(weights[0] * maps[0] + weights[1] * maps[1] + weights[2] * maps[2], 0.0, 1.0)

