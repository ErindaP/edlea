from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.alignment.coverage import analyze_pair_coverage, exclude_boundary_connected_detections
from src.alignment.keypoints import KeypointMatches
from src.types import DetectedChange


class FixedMatcher:
    name = "fixed"

    def __init__(self, source_points: np.ndarray, target_points: np.ndarray | None = None):
        self.source_points = source_points.astype(np.float32)
        self.target_points = (
            target_points.astype(np.float32) if target_points is not None else self.source_points.copy()
        )

    def match(self, image0: np.ndarray, image1: np.ndarray,
              image1_mask: np.ndarray | None = None) -> KeypointMatches:
        confidence = np.ones(len(self.source_points), dtype=np.float32)
        return KeypointMatches(self.source_points, self.target_points, confidence, self.name)


def point_grid(xs: list[float], ys: list[float]) -> np.ndarray:
    return np.float32([(x, y) for y in ys for x in xs])


def textured_image(seed: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(70, 210, (500, 640, 3), dtype=np.uint8)
    for index in range(35):
        center = (int(rng.integers(10, 630)), int(rng.integers(10, 490)))
        cv2.circle(image, center, 4 + index % 8, (25, 35, 45), 2)
    return image


def partial_view(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    source = np.float32([
        [0.10 * width, 15], [0.75 * width, 2],
        [0.78 * width, height - 15], [0.13 * width, height - 30],
    ])
    target = np.float32([[0, 0], [479, 0], [479, 459], [0, 459]])
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(source, target), (480, 460))


def test_pair_coverage_registers_partial_view_and_builds_comparison_mask():
    before = textured_image()
    after = partial_view(before)

    result = analyze_pair_coverage(before, after, matching_backend="opencv")

    assert result.alignment.success
    assert 45 < result.coverage_of_before_percent < 75
    assert 95 < result.comparable_after_percent <= 100
    assert result.alignment.valid_mask.shape == after.shape[:2]
    assert result.overlay.shape == after.shape
    assert result.wall_status.shape == before.shape[:2]
    assert set(np.unique(result.wall_status)) == {1, 2}
    assert result.alignment.num_inliers >= 10
    assert result.to_dict()["reference_mapping"] == "full_image_is_full_wall"


def test_pair_coverage_rejects_images_without_reliable_overlap():
    before = textured_image()
    after = np.full((460, 480, 3), 127, dtype=np.uint8)

    with pytest.raises(ValueError, match="correspondances géométriques"):
        analyze_pair_coverage(before, after, matching_backend="opencv")


def test_localized_but_stable_inliers_are_not_restricted_to_their_convex_hull():
    before = np.zeros((160, 200, 3), dtype=np.uint8)
    after = before.copy()
    # Reproduce the reported support: 16 inliers over 27% x 35% of the image.
    points = point_grid([55, 73, 91, 109], [40, 59, 77, 96])

    result = analyze_pair_coverage(before, after, matcher=FixedMatcher(points))

    assert result.support_mode == "stability_supported"
    assert result.alignment.num_inliers == 16
    assert result.to_dict()["source_span_percent"] == [27.0, 35.0]
    assert result.coverage_of_before_percent > 85
    assert result.comparable_after_percent > 85
    assert result.stability_models == 16
    assert result.median_reprojection_error < 0.01


def test_boundary_connected_changes_are_removed_without_dropping_interior_change():
    valid = np.zeros((100, 100), dtype=bool)
    valid[20:80, 20:80] = True
    boundary_component = np.zeros_like(valid)
    boundary_component[20:30, 35:45] = True
    interior_component = np.zeros_like(valid)
    interior_component[45:55, 45:55] = True
    detections = [
        DetectedChange(1, (35, 20, 45, 30), 100, 0.5, "wall", 0.01, boundary_component),
        DetectedChange(2, (45, 45, 55, 55), 100, 0.6, "wall", 0.01, interior_component),
    ]

    retained, rejected = exclude_boundary_connected_detections(detections, valid)

    assert rejected == 1
    assert len(retained) == 1
    assert retained[0].bbox == (45, 45, 55, 55)
    assert retained[0].id == 1
