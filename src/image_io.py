from __future__ import annotations

from pathlib import Path
from typing import BinaryIO
import io

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_image(source: str | Path | bytes | BinaryIO | Image.Image | np.ndarray,
               max_size: int | None = 1280) -> np.ndarray:
    """Load an image as RGB uint8, preserving its aspect ratio."""
    if isinstance(source, np.ndarray):
        image = source.copy()
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        image = image.astype(np.uint8)
    else:
        if isinstance(source, (str, Path)):
            pil_image = Image.open(source)
        elif isinstance(source, bytes):
            pil_image = Image.open(io.BytesIO(source))
        elif hasattr(source, "read"):
            pil_image = Image.open(source)
        elif isinstance(source, Image.Image):
            pil_image = source
        else:
            raise TypeError(f"Unsupported image source: {type(source)!r}")
        image = np.asarray(ImageOps.exif_transpose(pil_image).convert("RGB"))
        image = image.astype(np.uint8)
    if max_size and max(image.shape[:2]) > max_size:
        scale = max_size / max(image.shape[:2])
        size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(image)


def match_size(before: np.ndarray, after: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if before.shape[:2] == after.shape[:2]:
        return before, after
    return cv2.resize(before, (after.shape[1], after.shape[0]), interpolation=cv2.INTER_AREA), after
