from __future__ import annotations

import numpy as np

from src.comparison.fusion import compute_fused_change_map
from src.detection.classification import AnomalyClassifier
from src.detection.regions import detect_changes
from src.features.dinov3 import DinoFeatureExtractor
from src.housing.interactive import build_interactive_figure
from src.housing.plan import FloorPlan
from src.reporting.text import generate_text_report
from src.reporting.vision_language import build_visual_report_prompt, guard_visual_report
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


def test_thin_crack_survives_default_morphology():
    change = np.zeros((160, 160), dtype=np.float32)
    change[20:145, 78:80] = 0.75
    masks = HeuristicSegmenter().segment(np.zeros((160, 160, 3), dtype=np.uint8))
    detections = detect_changes(change, masks, threshold=0.3, min_area=100, morph_kernel=5, opening_kernel=0)
    assert detections
    assert detections[0].bbox[1] <= 20 and detections[0].bbox[3] >= 144


def test_hysteresis_extends_box_along_weak_crack_continuation():
    change = np.zeros((140, 100), dtype=np.float32)
    change[15:125, 49:51] = 0.18
    change[55:80, 48:52] = 0.85
    masks = HeuristicSegmenter().segment(np.zeros((140, 100, 3), dtype=np.uint8))

    detections = detect_changes(
        change, masks, threshold=0.5, min_area=20, morph_kernel=3,
        opening_kernel=0, hysteresis_ratio=0.3,
    )

    assert detections
    assert detections[0].bbox[1] <= 15
    assert detections[0].bbox[3] >= 124


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


def test_anomaly_classifier_uses_mask_elongation_inside_wide_box():
    before = np.zeros((160, 160, 3), dtype=np.uint8)
    after = before.copy()
    mask = np.zeros((160, 160), dtype=bool)
    for y in range(20, 140):
        x = 55 + y // 4
        mask[y, x:x + 3] = True
        after[y, x:x + 3] = 220
    detection = DetectedChange(1, (50, 20, 100, 140), int(mask.sum()), 0.6, "wall", 0.01, mask)

    classification = AnomalyClassifier().classify(before, after, mask.astype(np.float32), detection)

    assert classification.anomaly_type == "crack"
    assert classification.features["pca_elongation"] >= 4.0


def test_text_report_mentions_detected_change():
    text = generate_text_report({
        "changed_surface_ratio": 0.02,
        "detected_changes": [{"id": 1, "type": "impact", "surface": "wall", "confidence": 0.7,
                              "severity": "medium", "bbox": [1, 2, 10, 20], "area_pixels": 50,
                              "evidence": ["zone compacte"]}],
    })
    assert "impact probable" in text
    assert "Zone 1" in text


def test_text_report_appends_local_visual_analysis():
    text = generate_text_report({
        "max_change_score": 0.1,
        "detection_threshold": 0.3,
        "detected_changes": [],
        "global_visual_analysis": {
            "status": "success",
            "text": "Constat global : une fissure est visible sur les deux images.",
        },
    })

    assert "Analyse visuelle globale" in text
    assert "visible sur les deux images" in text


def test_visual_report_prompt_distinguishes_before_and_after():
    prompt = build_visual_report_prompt({"detected_changes": []})

    assert "première image" in prompt
    assert "AVANT" in prompt
    assert "seconde" in prompt
    assert "déjà visible dans les deux images" in prompt


def test_visual_report_guard_tempers_tiny_change_and_removes_depth_claims():
    guarded = guard_visual_report(
        "Image AVANT : fissure visible. Image APRÈS : fissure identique mais plus profonde.",
        {
            "changed_surface_ratio": 0.0005,
            "detected_changes": [{"type": "crack"}],
        },
    )

    assert "profon" not in guarded.lower()
    assert "visible dans les deux états" in guarded
    assert "ne permet pas de conclure" in guarded


def test_interactive_plan_exposes_clickable_anomaly_marker():
    plan = FloorPlan.sample_house()
    anomaly = {
        "id": 7,
        "type": "crack",
        "observation_id": "inspection_sortie",
        "location": {"wall_id": "living_north", "u": 0.4, "z_m": 1.3},
    }
    figure = build_interactive_figure(plan, [anomaly])
    marker_traces = [
        trace for trace in figure.data
        if list(getattr(trace, "customdata", []) or []) == ["anomaly:0"]
    ]

    assert len(marker_traces) == 1
    assert list(marker_traces[0].customdata) == ["anomaly:0"]
    assert figure.layout.scene.dragmode == "orbit"


def test_interactive_plan_exposes_clickable_walls():
    plan = FloorPlan.sample_house()
    figure = build_interactive_figure(plan)
    living_wall = next(trace for trace in figure.data if trace.name == "living_north")

    assert living_wall.meta == "wall:living_north"
    assert set(living_wall.customdata) == {"wall:living_north"}
