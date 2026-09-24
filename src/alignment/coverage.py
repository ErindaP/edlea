from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.types import AlignmentResult

from .keypoints import estimate_supported_homography, get_keypoint_matcher
from .warp import warp_image


@dataclass(frozen=True)
class PairCoverageResult:
    alignment: AlignmentResult
    coverage_of_before_percent: float
    comparable_after_percent: float
    matching_backend: str
    inlier_ratio: float
    overlay: np.ndarray
    wall_status: np.ndarray

    def to_dict(self) -> dict[str, float | int | str | bool]:
        return {
            "enabled": True,
            "coverage_of_before_percent": round(self.coverage_of_before_percent, 1),
            "comparable_after_percent": round(self.comparable_after_percent, 1),
            "matching_backend": self.matching_backend,
            "matches": self.alignment.num_matches,
            "inliers": self.alignment.num_inliers,
            "inlier_ratio": round(self.inlier_ratio, 3),
        }


def analyze_pair_coverage(
    before: np.ndarray,
    after: np.ndarray,
    *,
    matching_backend: str = "auto",
) -> PairCoverageResult:
    """Register After onto Before and retain only geometrically supported common pixels."""
    matcher = get_keypoint_matcher(matching_backend)
    # The generic matcher returns image0 -> image1, hence After -> Before here.
    matched = estimate_supported_homography(after, before, matcher)
    if matched is None:
        raise ValueError(
            "Les deux images ne présentent pas assez de correspondances géométriques fiables "
            "pour calculer leur couverture."
        )
    try:
        before_to_after = np.linalg.inv(matched.homography)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Le recalage de couverture est dégénéré.") from exc

    aligned_before, projected_before = warp_image(before, before_to_after, after.shape[:2])
    support_after = np.zeros(after.shape[:2], dtype=np.uint8)
    hull = cv2.convexHull(matched.inlier_source_points).astype(np.int32)
    cv2.fillConvexPoly(support_after, hull, 255)
    margin = max(3, int(min(after.shape[:2]) * 0.04))
    support_after = cv2.dilate(support_after, np.ones((margin, margin), np.uint8))
    comparable_after = projected_before & (support_after > 0)
    comparable_after = cv2.erode(comparable_after.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if int(comparable_after.sum()) < 100:
        raise ValueError("La zone commune fiable entre les deux images est trop petite pour être analysée.")

    support_in_before = cv2.warpPerspective(
        support_after,
        matched.homography,
        (before.shape[1], before.shape[0]),
        flags=cv2.INTER_NEAREST,
    ) > 0
    coverage_of_before = 100 * float(support_in_before.mean())
    comparable_after_percent = 100 * float(comparable_after.mean())
    wall_status = np.ones(before.shape[:2], dtype=np.uint8)
    wall_status[support_in_before] = 2

    overlay = after.copy()
    excluded = ~comparable_after
    tint = np.array([230, 145, 35], dtype=np.float32)
    overlay[excluded] = np.clip(0.35 * overlay[excluded] + 0.65 * tint, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(comparable_after.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (40, 210, 90), 3)

    alignment = AlignmentResult(
        aligned_before=aligned_before,
        homography=before_to_after,
        matches_before=matched.points_target,
        matches_after=matched.points_source,
        confidence=matched.confidence,
        inlier_mask=matched.inlier_mask,
        success=True,
        valid_mask=comparable_after,
    )
    return PairCoverageResult(
        alignment=alignment,
        coverage_of_before_percent=coverage_of_before,
        comparable_after_percent=comparable_after_percent,
        matching_backend=matched.backend,
        inlier_ratio=matched.inlier_ratio,
        overlay=overlay,
        wall_status=wall_status,
    )
