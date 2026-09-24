from __future__ import annotations

import cv2
import numpy as np

from ..types import DetectedChange, Mask


def regional_statistics(change_map: np.ndarray, masks: list[Mask], threshold: float,
                        analysis_mask: np.ndarray | None = None) -> list[dict]:
    result = []
    for region in masks:
        region_mask = region.binary_mask
        if analysis_mask is not None:
            region_mask = region_mask & analysis_mask
        values = change_map[region_mask]
        changed = values > threshold
        result.append({
            "surface": region.label,
            "mean_change": round(float(values.mean()) if values.size else 0.0, 6),
            "max_change": round(float(values.max()) if values.size else 0.0, 6),
            "changed_area_ratio": round(float(changed.mean()) if values.size else 0.0, 6),
        })
    return result


def detect_changes(change_map: np.ndarray, masks: list[Mask], threshold: float = 0.4,
                   min_area: int = 150, morph_kernel: int = 5, opening_kernel: int = 0,
                   hysteresis_ratio: float = 0.35,
                   analysis_mask: np.ndarray | None = None) -> list[DetectedChange]:
    """Detect strong changes, then grow them into connected low-confidence pixels.

    The two-threshold strategy keeps isolated low-score noise out while avoiding
    tiny boxes around long, thin defects whose score is not uniform everywhere.
    """
    strong = (change_map > threshold).astype(np.uint8)
    low_threshold = float(threshold) * float(np.clip(hysteresis_ratio, 0.05, 1.0))
    binary = (change_map > low_threshold).astype(np.uint8)
    valid = analysis_mask.astype(bool) if analysis_mask is not None else None
    if valid is not None:
        strong[~valid] = 0
        binary[~valid] = 0
    # An opening with a square 5x5 kernel erases thin cracks. Keep it disabled
    # by default and use the minimum-area filter for isolated noise instead.
    opening_size = int(opening_kernel)
    if opening_size > 1:
        opening = np.ones((opening_size, opening_size), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, opening)
        strong = cv2.morphologyEx(strong, cv2.MORPH_OPEN, opening)
    closing_size = int(morph_kernel)
    if closing_size > 1:
        closing = np.ones((closing_size, closing_size), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, closing)
        strong = cv2.morphologyEx(strong, cv2.MORPH_CLOSE, closing)
    if valid is not None:
        strong[~valid] = 0
        binary[~valid] = 0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    image_area = int(valid.sum()) if valid is not None else change_map.shape[0] * change_map.shape[1]
    detections: list[DetectedChange] = []
    next_id = 1
    for label_id in range(1, num_labels):
        component = labels == label_id
        strong_component = np.logical_and(component, strong.astype(bool))
        strong_area = int(strong_component.sum())
        if strong_area < min_area:
            continue
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        score = float(change_map[strong_component].mean())
        surface, best_overlap = "other", 0
        for region in masks:
            overlap = int(np.logical_and(component, region.binary_mask).sum())
            if overlap > best_overlap:
                surface, best_overlap = region.label, overlap
        x, y, w, h = [int(value) for value in stats[label_id, :4]]
        detections.append(DetectedChange(next_id, (x, y, x + w, y + h), area, score, surface,
                                          area / max(1, image_area), binary_mask=component))
        next_id += 1
    return sorted(detections, key=lambda item: item.score, reverse=True)
