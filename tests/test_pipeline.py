from __future__ import annotations

import numpy as np

from src.comparison.fusion import compute_fused_change_map
from src.detection.classification import AnomalyClassifier
from src.detection.regions import detect_changes
from src.features.dinov3 import DinoFeatureExtractor
from src.reporting.text import generate_text_report
from src.segmentation.segmenter import HeuristicSegmenter
from src.types import DetectedChange


def test_identical_features_have_zero_change():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[20:80, 30:90] = [120, 80, 40]
    extractor = DinoFeatureExtractor(model_name="unused", backend="fallback")
    features = extractor.extract(image)
    dino = 1.0 - np.sum(features * features, axis=-1)
    assert float(np.max(np.abs(dino))) < 1e-5


def test_fusion_weights_are_normalized():
    output = compute_fused_change_map(np.ones((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), 6, 3, 1)
    assert np.allclose(output, 0.6)


def test_black_line_is_localized():
    change = np.zeros((100, 100), dtype=np.float32)
    change[45:55, 20:80] = 0.9
    masks = HeuristicSegmenter().segment(np.zeros((100, 100, 3), dtype=np.uint8))
    detections = detect_changes(change, masks, threshold=0.4, min_area=10, morph_kernel=3)
    assert detections
    assert detections[0].bbox[0] <= 20 and detections[0].bbox[2] >= 79


def test_anomaly_classifier_returns_explainable_type():
    before = np.zeros((100, 160, 3), dtype=np.uint8)
    after = before.copy()
    after[45:50, 20:140] = 220
    mask = np.zeros((100, 160), dtype=bool)
    mask[45:50, 20:140] = True
    detection = DetectedChange(1, (20, 45, 140, 50), int(mask.sum()), 0.8, "wall", 0.03, mask)
    classification = AnomalyClassifier().classify(before, after, mask.astype(np.float32), detection)
    assert classification.anomaly_type == "crack"
    assert classification.evidence
    assert 0.0 <= classification.confidence <= 1.0


def test_text_report_mentions_detected_change():
    text = generate_text_report({
        "changed_surface_ratio": 0.02,
        "detected_changes": [{"id": 1, "type": "impact", "surface": "wall", "confidence": 0.7,
                              "severity": "medium", "bbox": [1, 2, 10, 20], "area_pixels": 50,
                              "evidence": ["zone compacte"]}],
    })
    assert "impact probable" in text
    assert "Zone 1" in text
