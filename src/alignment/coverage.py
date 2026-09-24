from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.types import AlignmentResult, DetectedChange

from .keypoints import HomographyMatch, KeypointMatcher, estimate_supported_homography, get_keypoint_matcher
from .warp import warp_image


STABILITY_TOLERANCE_RATIO = 0.025
STABILITY_PERCENTILE = 90
MAX_STABILITY_MODELS = 32


def _inlier_reprojection_errors(matched: HomographyMatch) -> np.ndarray:
    source = matched.inlier_source_points
    target = matched.points_target[matched.inlier_mask]
    projected = cv2.perspectiveTransform(source[:, None, :], matched.homography)[:, 0, :]
    return np.linalg.norm(projected - target, axis=1)


def _stability_supported_mask(
    matched: HomographyMatch,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    projected_target: np.ndarray,
) -> tuple[np.ndarray | None, float, int]:
    """Estimate where a locally supported homography remains stable when extrapolated.

    A leave-one-out family exposes homographies that fit the same local inliers but
    diverge away from them. Unlike a convex hull, this retains textureless wall
    areas when the projective model is stable there.
    """
    source = matched.inlier_source_points
    target = matched.points_target[matched.inlier_mask]
    if len(source) < 10:
        return None, 0.0, 0

    height, width = source_shape
    grid_step = max(4, round(min(source_shape) / 80))
    grid_width = max(2, int(np.ceil(width / grid_step)))
    grid_height = max(2, int(np.ceil(height / grid_step)))
    xs = np.linspace(0, width - 1, grid_width, dtype=np.float32)
    ys = np.linspace(0, height - 1, grid_height, dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 1, 2)
    baseline = cv2.perspectiveTransform(grid, matched.homography)[:, 0, :]

    omitted_indices = np.linspace(
        0,
        len(source) - 1,
        min(len(source), MAX_STABILITY_MODELS),
        dtype=int,
    )
    deviations: list[np.ndarray] = []
    for omitted in np.unique(omitted_indices):
        candidate, _ = cv2.findHomography(
            np.delete(source, omitted, axis=0),
            np.delete(target, omitted, axis=0),
            0,
        )
        if candidate is None or not np.all(np.isfinite(candidate)):
            continue
        prediction = cv2.perspectiveTransform(grid, candidate)[:, 0, :]
        if np.all(np.isfinite(prediction)):
            deviations.append(np.linalg.norm(prediction - baseline, axis=1))
    if len(deviations) < 4:
        return None, 0.0, len(deviations)

    uncertainty = np.percentile(np.stack(deviations), STABILITY_PERCENTILE, axis=0)
    uncertainty = uncertainty.reshape(grid_height, grid_width).astype(np.float32)
    uncertainty = cv2.resize(uncertainty, (width, height), interpolation=cv2.INTER_LINEAR)
    tolerance = max(8.0, min(target_shape) * STABILITY_TOLERANCE_RATIO)
    stable = (uncertainty <= tolerance) & projected_target

    close_radius = max(3, round(min(source_shape) * 0.015))
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * close_radius + 1, 2 * close_radius + 1),
    )
    stable = cv2.morphologyEx(stable.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel).astype(bool)
    stable &= projected_target

    # Discard any disconnected stable island that is unrelated to the actual
    # inlier support. Such islands can be mathematical coincidences far away.
    seed = np.zeros(source_shape, dtype=np.uint8)
    cv2.fillConvexPoly(seed, cv2.convexHull(source).astype(np.int32), 1)
    labels_count, labels = cv2.connectedComponents(stable.astype(np.uint8), 8)
    supported = np.zeros_like(stable)
    for label_id in range(1, labels_count):
        component = labels == label_id
        if np.any(component & (seed > 0)):
            supported |= component
    minimum_area = max(100, round(float(projected_target.sum()) * 0.05))
    if int(supported.sum()) < minimum_area:
        return None, tolerance, len(deviations)
    return supported, tolerance, len(deviations)


def exclude_boundary_connected_detections(
    detections: list[DetectedChange],
    analysis_mask: np.ndarray,
    *,
    band_pixels: int = 2,
) -> tuple[list[DetectedChange], int]:
    """Drop changes clipped by the boundary of a coverage-analysis mask."""
    radius = max(1, int(band_pixels))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    valid = analysis_mask.astype(bool)
    interior = cv2.erode(
        valid.astype(np.uint8),
        kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    boundary = valid & ~interior
    retained = [
        detection for detection in detections
        if detection.binary_mask is None or not np.any(detection.binary_mask & boundary)
    ]
    for identifier, detection in enumerate(retained, start=1):
        detection.id = identifier
    return retained, len(detections) - len(retained)


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
    median_reprojection_error: float
    p95_reprojection_error: float
    stability_threshold_pixels: float
    stability_models: int
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
            "median_reprojection_error_pixels": round(self.median_reprojection_error, 2),
            "p95_reprojection_error_pixels": round(self.p95_reprojection_error, 2),
            "stability_threshold_pixels": round(self.stability_threshold_pixels, 1),
            "stability_models": self.stability_models,
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
    reprojection_errors = _inlier_reprojection_errors(matched)

    # The full Before frame still represents the full selected wall. Localized
    # inliers do not imply that every extrapolation is invalid: use jackknife
    # stability to retain textureless wall areas where the model remains sound.
    support_after = projected_before.copy()
    support_mode = "full_footprint"
    stability_threshold = 0.0
    stability_models = 0
    if not broadly_supported:
        stable_support, stability_threshold, stability_models = _stability_supported_mask(
            matched,
            after.shape[:2],
            before.shape[:2],
            projected_before,
        )
        if stable_support is not None:
            support_mode = "stability_supported"
            support_after = stable_support
        else:
            support_mode = "inlier_supported"
            support_after = np.zeros(after.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(support_after, cv2.convexHull(inlier_source).astype(np.int32), 1)
            support_radius = max(5, round(min(after.shape[:2]) * 0.06))
            support_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * support_radius + 1, 2 * support_radius + 1),
            )
            support_after = cv2.dilate(support_after, support_kernel).astype(bool)

    comparable_after = projected_before & (support_after > 0)
    safety_margin = max(3, round(min(after.shape[:2]) * 0.0075))
    safety_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * safety_margin + 1, 2 * safety_margin + 1),
    )
    comparable_after = cv2.erode(
        comparable_after.astype(np.uint8),
        safety_kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
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
        median_reprojection_error=float(np.median(reprojection_errors)),
        p95_reprojection_error=float(np.percentile(reprojection_errors, 95)),
        stability_threshold_pixels=stability_threshold,
        stability_models=stability_models,
        overlay=overlay,
        wall_status=wall_status,
    )
