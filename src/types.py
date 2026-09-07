from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class AlignmentResult:
    aligned_before: np.ndarray
    homography: np.ndarray
    matches_before: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))
    matches_after: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))
    confidence: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float32))
    inlier_mask: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=bool))
    success: bool = False
    valid_mask: np.ndarray | None = None

    @property
    def num_matches(self) -> int:
        return int(len(self.matches_before))

    @property
    def num_inliers(self) -> int:
        return int(self.inlier_mask.sum()) if self.inlier_mask.size else 0


@dataclass
class Mask:
    id: int
    label: str
    binary_mask: np.ndarray


@dataclass
class DetectedChange:
    id: int
    bbox: tuple[int, int, int, int]
    area: int
    score: float
    surface: str
    changed_area_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "surface": self.surface,
            "bbox": list(self.bbox),
            "score": round(self.score, 6),
            "area_pixels": self.area,
            "changed_area_ratio": round(self.changed_area_ratio, 6),
        }

