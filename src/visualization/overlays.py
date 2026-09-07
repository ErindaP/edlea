from __future__ import annotations

import cv2
import numpy as np

from ..types import AlignmentResult, DetectedChange


def render_heatmap(change_map: np.ndarray, base: np.ndarray | None = None) -> np.ndarray:
    heat = cv2.applyColorMap(np.uint8(np.clip(change_map, 0, 1) * 255), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(base, 0.45, heat, 0.55, 0) if base is not None else heat


def render_detections(image: np.ndarray, detections: list[DetectedChange]) -> np.ndarray:
    canvas = image.copy()
    for item in detections:
        x1, y1, x2, y2 = item.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (235, 45, 45), 3)
        label = f"#{item.id} {item.surface} {item.score:.2f} ({item.changed_area_ratio:.1%})"
        cv2.putText(canvas, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 45, 45), 2, cv2.LINE_AA)
    return canvas


def render_matches(before: np.ndarray, after: np.ndarray, alignment: AlignmentResult) -> np.ndarray:
    left = cv2.cvtColor(before, cv2.COLOR_RGB2BGR)
    right = cv2.cvtColor(after, cv2.COLOR_RGB2BGR)
    key_before = [cv2.KeyPoint(float(x), float(y), 5) for x, y in alignment.matches_before]
    key_after = [cv2.KeyPoint(float(x), float(y), 5) for x, y in alignment.matches_after]
    matches = [cv2.DMatch(i, i, 0, 0) for i in range(alignment.num_matches)]
    canvas = cv2.drawMatches(left, key_before, right, key_after, matches, None,
                             matchesMask=alignment.inlier_mask.astype(int).tolist() if alignment.inlier_mask.size else None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

