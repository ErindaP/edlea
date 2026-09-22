from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..types import DetectedChange


@dataclass
class AnomalyClassification:
    anomaly_type: str
    confidence: float
    severity: str
    evidence: list[str]
    features: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.anomaly_type,
            "confidence": round(float(self.confidence), 4),
            "severity": self.severity,
            "evidence": self.evidence,
            "features": {key: round(float(value), 6) for key, value in self.features.items()},
        }


def _crop(image: np.ndarray, bbox: tuple[int, int, int, int], padding: int = 8) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    height, width = image.shape[:2]
    return image[max(0, y1 - padding):min(height, y2 + padding), max(0, x1 - padding):min(width, x2 + padding)]


def extract_anomaly_features(before: np.ndarray, after: np.ndarray, change_map: np.ndarray,
                             detection: DetectedChange) -> dict[str, float]:
    x1, y1, x2, y2 = detection.bbox
    width, height = max(1, x2 - x1), max(1, y2 - y1)
    crop_before = _crop(before, detection.bbox)
    crop_after = _crop(after, detection.bbox)
    crop_change = change_map[y1:y2, x1:x2]
    mask = detection.binary_mask[y1:y2, x1:x2] if detection.binary_mask is not None else crop_change > 0
    gray = cv2.cvtColor(crop_after, cv2.COLOR_RGB2GRAY) if crop_after.size else np.zeros((1, 1), np.uint8)
    edges = cv2.Canny(gray, 80, 160) if gray.size else np.zeros_like(gray)
    mask_area = max(1, int(mask.sum()))
    perimeter = float(cv2.arcLength(cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0], True)) if np.any(mask) else 0.0
    compactness = float(4.0 * np.pi * mask_area / max(perimeter * perimeter, 1.0))
    changed_pixels = crop_change[mask] if np.any(mask) else crop_change.ravel()
    before_gray = cv2.cvtColor(crop_before, cv2.COLOR_RGB2GRAY) if crop_before.size else gray
    after_gray = cv2.cvtColor(crop_after, cv2.COLOR_RGB2GRAY) if crop_after.size else gray
    gray_delta = np.abs(after_gray.astype(np.float32) - before_gray.astype(np.float32)) / 255.0
    texture = float(np.std(after_gray.astype(np.float32)) / 255.0) if after_gray.size else 0.0
    edge_density = float(edges.mean() / 255.0) if edges.size else 0.0
    area_ratio_bbox = mask_area / float(width * height)
    aspect_ratio = max(width, height) / float(min(width, height))
    coordinates = np.column_stack(np.nonzero(mask)).astype(np.float32)
    if len(coordinates) >= 3:
        eigenvalues = np.linalg.eigvalsh(np.cov(coordinates, rowvar=False))
        pca_elongation = float(np.sqrt(max(eigenvalues[-1], 0.0) / max(eigenvalues[0], 1e-6)))
    else:
        pca_elongation = 1.0
    feature_change = float(np.mean(changed_pixels)) if changed_pixels.size else float(np.mean(crop_change))
    return {
        "bbox_width": float(width),
        "bbox_height": float(height),
        "aspect_ratio": float(aspect_ratio),
        "pca_elongation": pca_elongation,
        "mask_fill_ratio": float(area_ratio_bbox),
        "compactness": float(np.clip(compactness, 0.0, 1.0)),
        "feature_change": feature_change,
        "pixel_change": float(np.mean(gray_delta)) if gray_delta.size else 0.0,
        "texture": texture,
        "edge_density": edge_density,
        "area_ratio_image": float(detection.changed_area_ratio),
    }


class AnomalyClassifier:
    """Explainable V2 classifier; conservative rules leave ambiguous cases unknown."""

    def classify(self, before: np.ndarray, after: np.ndarray, change_map: np.ndarray,
                 detection: DetectedChange) -> AnomalyClassification:
        features = extract_anomaly_features(before, after, change_map, detection)
        aspect = features["aspect_ratio"]
        elongation = max(aspect, features["pca_elongation"])
        fill = features["mask_fill_ratio"]
        compactness = features["compactness"]
        feature_change = features["feature_change"]
        evidence: list[str] = []
        candidates: list[tuple[str, float, list[str]]] = []

        if elongation >= 4.0 and (fill <= 0.65 or detection.score >= 0.65):
            candidates.append(("crack", 0.60 + min(0.22, (elongation - 4.0) * 0.04),
                               ["distribution très allongée des pixels modifiés", "zone peu remplie dans sa bounding box"]))
        if compactness >= 0.45 and fill >= 0.45 and feature_change >= 0.45:
            candidates.append(("impact", 0.55 + min(0.25, feature_change * 0.25), ["zone compacte", "contraste local important"]))
        if compactness >= 0.20 and fill >= 0.35 and features["texture"] <= 0.35:
            candidates.append(("stain_or_dirt", 0.52 + min(0.20, feature_change * 0.20), ["variation diffuse de texture/couleur"]))
        if fill >= 0.75 and feature_change >= 0.50 and aspect <= 3.0:
            candidates.append(("object_change", 0.52 + min(0.22, feature_change * 0.20), ["zone largement remplie", "modification visuelle étendue"]))
        if features["texture"] >= 0.38 and features["edge_density"] >= 0.08 and fill >= 0.25:
            candidates.append(("paint_peeling", 0.50 + min(0.18, features["texture"] * 0.20), ["texture irrégulière", "contours fragmentés"]))

        if not candidates:
            anomaly_type, confidence = "unknown", 0.35
            evidence = ["indices visuels insuffisants pour une classification fiable"]
        else:
            anomaly_type, confidence, evidence = max(candidates, key=lambda item: item[1])
            confidence = min(0.90, confidence)
        severity = self._severity(detection, confidence)
        return AnomalyClassification(anomaly_type, confidence, severity, evidence, features)

    @staticmethod
    def _severity(detection: DetectedChange, confidence: float) -> str:
        if detection.changed_area_ratio >= 0.05 or detection.score >= 0.80:
            return "high"
        if detection.changed_area_ratio >= 0.01 or confidence >= 0.70:
            return "medium"
        return "low"
