from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.alignment.coverage import analyze_pair_coverage


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
