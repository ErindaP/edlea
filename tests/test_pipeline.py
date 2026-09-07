from __future__ import annotations

import numpy as np

from src.comparison.fusion import compute_fused_change_map
from src.detection.regions import detect_changes
from src.features.dinov3 import DinoFeatureExtractor
from src.segmentation.segmenter import HeuristicSegmenter


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

