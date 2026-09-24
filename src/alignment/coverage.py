from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.types import AlignmentResult

from .keypoints import KeypointMatcher, estimate_supported_homography, get_keypoint_matcher
from .warp import warp_image


@dataclass(frozen=True)
class PairCoverageResult:
    alignment: AlignmentResult
    coverage_of_before_percent: float
    comparable_after_percent: float
    matching_backend: str
    inlier_ratio: float
    support_mode: str
    source_span: tuple[float, float]
    target_span: tuple[float, float]
    safety_margin_pixels: int
    overlay: np.ndarray
    wall_status: np.ndarray

    def to_dict(self) -> dict[str, float | int | str | bool]:
        return {
            "enabled": True,
            "reference_mapping": "full_image_is_full_wall",
            "coverage_of_before_percent": round(self.coverage_of_before_percent, 1),
            "comparable_after_percent": round(self.comparable_after_percent, 1),
            "matching_backend": self.matching_backend,
            "matches": self.alignment.num_matches,
            "inliers": self.alignment.num_inliers,
            "inlier_ratio": round(self.inlier_ratio, 3),
            "support_mode": self.support_mode,
            "source_span_percent": [round(100 * value, 1) for value in self.source_span],
            "target_span_percent": [round(100 * value, 1) for value in self.target_span],
            "safety_margin_pixels": self.safety_margin_pixels,
        }


def analyze_pair_coverage(
    before: np.ndarray,
    after: np.ndarray,
    *,
    matching_backend: str = "auto",
    matcher: KeypointMatcher | None = None,
) -> PairCoverageResult:
    """Register After onto Before and retain only geometrically supported common pixels."""
    matcher = matcher or get_keypoint_matcher(matching_backend)
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
    inlier_source = matched.inlier_source_points
    inlier_target = matched.points_target[matched.inlier_mask]
    source_span_array = np.ptp(inlier_source, axis=0) / np.array([after.shape[1], after.shape[0]])
    target_span_array = np.ptp(inlier_target, axis=0) / np.array([before.shape[1], before.shape[0]])
    source_span = (float(source_span_array[0]), float(source_span_array[1]))
    target_span = (float(target_span_array[0]), float(target_span_array[1]))
    broadly_supported = (
        matched.inliers >= 30
        and min(source_span) >= 0.35
        and min(target_span) >= 0.35
    )

    # The full Before frame still represents the full selected wall. However,
    # a homography supported by a small feature cluster must not be extrapolated
    # all the way to the image borders: those warped borders look like defects.
    support_after = np.full(after.shape[:2], 255, dtype=np.uint8)
    support_mode = "full_footprint"
    if not broadly_supported:
        support_mode = "inlier_supported"
        support_after.fill(0)
        cv2.fillConvexPoly(support_after, cv2.convexHull(inlier_source).astype(np.int32), 255)
        support_radius = max(5, round(min(after.shape[:2]) * 0.06))
        support_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * support_radius + 1, 2 * support_radius + 1),
        )
        support_after = cv2.dilate(support_after, support_kernel)

    comparable_after = projected_before & (support_after > 0)
    safety_margin = max(3, round(min(after.shape[:2]) * (0.01 if broadly_supported else 0.005)))
    safety_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * safety_margin + 1, 2 * safety_margin + 1),
    )
    comparable_after = cv2.erode(comparable_after.astype(np.uint8), safety_kernel).astype(bool)
    if int(comparable_after.sum()) < 100:
        raise ValueError("La zone commune fiable entre les deux images est trop petite pour être analysée.")

    support_in_before = cv2.warpPerspective(
        comparable_after.astype(np.uint8) * 255,
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
        support_mode=support_mode,
        source_span=source_span,
        target_span=target_span,
        safety_margin_pixels=safety_margin,
        overlay=overlay,
        wall_status=wall_status,
    )
