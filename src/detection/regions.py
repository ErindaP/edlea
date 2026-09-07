from __future__ import annotations

import cv2
import numpy as np

from ..types import DetectedChange, Mask


def regional_statistics(change_map: np.ndarray, masks: list[Mask], threshold: float) -> list[dict]:
    result = []
    for region in masks:
        values = change_map[region.binary_mask]
        changed = values > threshold
        result.append({
            "surface": region.label,
            "mean_change": round(float(values.mean()) if values.size else 0.0, 6),
            "max_change": round(float(values.max()) if values.size else 0.0, 6),
            "changed_area_ratio": round(float(changed.mean()) if values.size else 0.0, 6),
        })
    return result


def detect_changes(change_map: np.ndarray, masks: list[Mask], threshold: float = 0.4,
                   min_area: int = 150, morph_kernel: int = 5) -> list[DetectedChange]:
    binary = (change_map > threshold).astype(np.uint8)
    kernel_size = max(1, int(morph_kernel))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    image_area = change_map.shape[0] * change_map.shape[1]
    detections: list[DetectedChange] = []
    next_id = 1
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component = labels == label_id
        score = float(change_map[component].mean())
        surface, best_overlap = "other", 0
        for region in masks:
            overlap = int(np.logical_and(component, region.binary_mask).sum())
            if overlap > best_overlap:
                surface, best_overlap = region.label, overlap
        x, y, w, h = [int(value) for value in stats[label_id, :4]]
        detections.append(DetectedChange(next_id, (x, y, x + w, y + h), area, score, surface,
                                          area / image_area, binary_mask=component))
        next_id += 1
    return sorted(detections, key=lambda item: item.score, reverse=True)
