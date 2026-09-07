from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..types import Mask


class Segmenter(ABC):
    @abstractmethod
    def segment(self, image: np.ndarray) -> list[Mask]:
        raise NotImplementedError


class HeuristicSegmenter(Segmenter):
    """Stable V1 region masks, ready to be replaced by SAM later."""

    def segment(self, image: np.ndarray) -> list[Mask]:
        height, width = image.shape[:2]
        y = np.arange(height)[:, None]
        floor_start = int(height * 0.72)
        wall = np.broadcast_to(y < floor_start, (height, width)).copy()
        floor = np.broadcast_to(y >= floor_start, (height, width)).copy()
        return [Mask(1, "wall", wall), Mask(2, "floor", floor)]

