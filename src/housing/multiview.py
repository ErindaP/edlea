"""Reference-relative wall coverage from unordered, overlapping photographs.

Each reference is calibrated onto a rectangular portion of one planar wall.
New views are registered to a reference, then projected into that wall atlas.
The result is intentionally conservative: an unregistered view counts as nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.comparison.pixel import compute_pixel_maps
from src.detection.regions import detect_changes
from src.alignment.keypoints import KeypointMatcher, estimate_supported_homography, get_keypoint_matcher
from src.image_io import load_image
from src.segmentation.segmenter import HeuristicSegmenter
from src.visualization.overlays import render_detections, render_heatmap

from .plan import FloorPlan, Wall


ATLAS_WIDTH = 640


@dataclass(frozen=True)
class ReferenceView:
    id: str
    wall_id: str
    image: np.ndarray
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    image_quad: tuple[tuple[float, float], ...] = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))


def validate_reference(bounds: tuple[float, ...], quad: tuple[tuple[float, float], ...]) -> None:
    if len(bounds) != 4 or not all(np.isfinite(bounds)):
        raise ValueError("Les limites du mur doivent contenir quatre nombres finis.")
    u0, v0, u1, v1 = bounds
    if not (0 <= u0 < u1 <= 1 and 0 <= v0 < v1 <= 1):
        raise ValueError("Les limites doivent respecter 0 ≤ u0 < u1 ≤ 1 et 0 ≤ v0 < v1 ≤ 1.")
    if len(quad) != 4 or any(len(point) != 2 or not all(np.isfinite(point)) or
                             not all(0 <= value <= 1 for value in point) for point in quad):
        raise ValueError("Indiquez quatre coins (x,y) normalisés entre 0 et 1.")
    polygon = np.float32(quad)
    if not cv2.isContourConvex(polygon) or abs(cv2.contourArea(polygon)) < 0.01:
        raise ValueError("Les coins doivent former un quadrilatère convexe non dégénéré.")


def atlas_size(wall: Wall) -> tuple[int, int]:
    return ATLAS_WIDTH, int(np.clip(round(ATLAS_WIDTH * wall.height / max(wall.length, 0.1)), 320, 900))


def _reference_projection(view: ReferenceView, wall: Wall) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    validate_reference(view.bounds, view.image_quad)
    width, height = atlas_size(wall)
    image_height, image_width = view.image.shape[:2]
    src = np.float32([(x * (image_width - 1), y * (image_height - 1)) for x, y in view.image_quad])
    u0, v0, u1, v1 = view.bounds
    dst = np.float32([[u0 * (width - 1), v0 * (height - 1)],
                      [u1 * (width - 1), v0 * (height - 1)],
                      [u1 * (width - 1), v1 * (height - 1)],
                      [u0 * (width - 1), v1 * (height - 1)]])
    projection = cv2.getPerspectiveTransform(src, dst)
    source_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    cv2.fillConvexPoly(source_mask, np.round(src).astype(np.int32), 255)
    return projection, source_mask, (width, height)


def _warp(image: np.ndarray, projection: np.ndarray, size: tuple[int, int],
          source_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    width, height = size
    if source_mask is None:
        source_mask = np.full(image.shape[:2], 255, dtype=np.uint8)
    warped = cv2.warpPerspective(image, projection, (width, height), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(source_mask, projection, (width, height), flags=cv2.INTER_NEAREST) > 0
    return warped, mask


def _match(scan: np.ndarray, reference: ReferenceView, reference_mask: np.ndarray,
           matcher: KeypointMatcher) -> dict | None:
    """Estimate scan->reference homography and reject weak/degenerate matches."""
    matched = estimate_supported_homography(scan, reference.image, matcher, reference_mask)
    if matched is None:
        return None
    return {"homography": matched.homography, "inliers": matched.inliers, "matches": matched.matches,
            "inlier_ratio": round(matched.inlier_ratio, 3),
            "inlier_points": matched.inlier_source_points,
            "mean_confidence": round(float(matched.confidence[matched.inlier_mask].mean()), 3),
            "matching_backend": matched.backend}


def analyze_scan(plan: FloorPlan, references: list[ReferenceView], scans: list[tuple[str, np.ndarray]],
                 *, threshold: float = 0.32, min_area: int = 90,
                 matching_backend: str = "auto") -> dict:
    """Compute per-wall coverage and indicative changes, independently of photo count."""
    if not references:
        raise ValueError("Ajoutez au moins une photo de référence calibrée.")
    matcher = get_keypoint_matcher(matching_backend)
    prepared = []
    wall_data = {}
    for reference in references:
        wall = plan.wall(reference.wall_id)
        image = load_image(reference.image, 1280)
        view = ReferenceView(reference.id, reference.wall_id, image, reference.bounds, reference.image_quad)
        projection, source_mask, size = _reference_projection(view, wall)
        warped, mask = _warp(image, projection, size, source_mask)
        item = wall_data.setdefault(wall.id, {
            "wall": wall, "size": size, "reference_image": np.zeros_like(warped),
            "reference_mask": np.zeros(mask.shape, dtype=bool),
            "scan_image": np.zeros_like(warped), "scan_mask": np.zeros(mask.shape, dtype=bool),
            "change_map": np.zeros(mask.shape, dtype=np.float32),
            "change_source": np.full(mask.shape, -1, dtype=np.int32), "reference_ids": [],
        })
        new_pixels = mask & ~item["reference_mask"]
        item["reference_image"][new_pixels] = warped[new_pixels]
        item["reference_mask"] |= mask
        item["reference_ids"].append(view.id)
        prepared.append((view, projection, source_mask, size))

    registrations = []
    scan_ids = [scan_id for scan_id, _ in scans]
    for scan_index, (scan_id, source) in enumerate(scans):
        scan = load_image(source, 1280)
        candidates = []
        for view, projection, source_mask, size in prepared:
            match = _match(scan, view, source_mask, matcher)
            if match is not None:
                candidates.append((match["inliers"], view, projection, size, match))
        if not candidates:
            registrations.append({"image_id": scan_id, "status": "non_localisee", "wall_id": None,
                                  "reference_candidates": []})
            continue
        candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        _, view, projection, size, match = candidates[0]
        candidate_rows = [{"reference_id": candidate_view.id, "wall_id": candidate_view.wall_id,
                           "inliers": int(candidate_match["inliers"]),
                           "matches": int(candidate_match["matches"]),
                           "inlier_ratio": float(candidate_match["inlier_ratio"]),
                           "mean_confidence": float(candidate_match["mean_confidence"]),
                           "matching_backend": candidate_match["matching_backend"],
                           "relative_score_percent": round(100 * candidate_match["inliers"] /
                                                           max(1, match["inliers"]), 1)}
                          for _, candidate_view, _, _, candidate_match in candidates[:3]]
        if any(other_view.wall_id != view.wall_id and score >= match["inliers"] * 0.85
               for score, other_view, *_ in candidates[1:]):
            registrations.append({"image_id": scan_id, "status": "mur_ambigu", "wall_id": None,
                                  "reference_candidates": candidate_rows})
            continue
        image_to_atlas = projection @ match["homography"]
        # Do not extrapolate coverage far beyond the matched planar features.
        # This is especially important when a nearly blank wall offers only a
        # few keypoints on a fixture occupying a small part of the photograph.
        supported = np.zeros(scan.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(supported, cv2.convexHull(match["inlier_points"]).astype(np.int32), 255)
        margin = max(3, int(min(scan.shape[:2]) * 0.04))
        supported = cv2.dilate(supported, np.ones((margin, margin), np.uint8))
        warped, mask = _warp(scan, image_to_atlas, size, supported)
        item = wall_data[view.wall_id]
        # Count only the portions for which a reference exists; extrapolation
        # beyond known reference coverage cannot increase the denominator.
        mask &= item["reference_mask"]
        if int(mask.sum()) < 100:
            registrations.append({"image_id": scan_id, "status": "non_localisee", "wall_id": None,
                                  "reference_candidates": candidate_rows})
            continue
        new_pixels = mask & ~item["scan_mask"]
        item["scan_image"][new_pixels] = warped[new_pixels]
        # Every registered view contributes evidence, even if another photo
        # already covered the same cells. Otherwise a change visible only in
        # the second view would be hidden by the first-wins mosaic.
        comparable = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        pixel_maps = compute_pixel_maps(item["reference_image"], warped)
        view_change = (0.65 * pixel_maps["ssim"] + 0.35 * pixel_maps["rgb"]).astype(np.float32)
        stronger = comparable & (view_change > item["change_map"])
        item["scan_image"][stronger] = warped[stronger]
        item["change_map"][stronger] = view_change[stronger]
        item["change_source"][stronger] = scan_index
        item["scan_mask"] |= mask
        registrations.append({"image_id": scan_id, "status": "localisee", "wall_id": view.wall_id,
                              "reference_id": view.id, "inliers": match["inliers"],
                              "matches": match["matches"], "inlier_ratio": match["inlier_ratio"],
                              "mean_confidence": match["mean_confidence"],
                              "matching_backend": match["matching_backend"],
                              "coverage_of_reference_percent": round(float(100 * mask.sum() /
                                                                            max(1, item["reference_mask"].sum())), 1),
                              "reference_candidates": candidate_rows})

    walls = {}
    total_reference_area = 0.0
    total_scanned_area = 0.0
    for wall_id, item in wall_data.items():
        reference_mask = item["reference_mask"]
        scan_mask = item["scan_mask"]
        overlap = reference_mask & scan_mask
        wall = item["wall"]
        wall_area = wall.length * wall.height
        reference_area = wall_area * float(reference_mask.mean())
        scanned_area = wall_area * float(overlap.mean())
        total_reference_area += reference_area
        total_scanned_area += scanned_area
        status = np.zeros(reference_mask.shape, dtype=np.uint8)
        status[reference_mask] = 1
        status[overlap] = 2
        before = item["reference_image"]
        after = item["scan_image"]
        change_map = item["change_map"]
        detections = detect_changes(change_map, HeuristicSegmenter().segment(after),
                                    threshold=threshold, min_area=min_area, morph_kernel=3,
                                    analysis_mask=overlap)
        changes = []
        width, height = item["size"]
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            source_values = item["change_source"][detection.binary_mask]
            source_values = source_values[source_values >= 0]
            source_image_ids = []
            if source_values.size:
                counts = np.bincount(source_values, minlength=len(scan_ids))
                source_image_ids = [scan_ids[index] for index in np.argsort(counts)[::-1]
                                    if counts[index] > 0][:3]
            changes.append({"id": detection.id, "bbox": list(detection.bbox),
                            "score": round(detection.score, 3),
                            "u": round((x1 + x2) / (2 * width), 4),
                            "v": round((y1 + y2) / (2 * height), 4),
                            "source_image_ids": source_image_ids})
        walls[wall_id] = {
            "reference_area_m2": round(reference_area, 3), "scanned_area_m2": round(scanned_area, 3),
            "coverage_percent": round(100 * scanned_area / reference_area, 1) if reference_area else 0.0,
            "reference_ids": item["reference_ids"], "changes": changes, "status": status, "reference_atlas": before,
            "scan_atlas": after, "change_heatmap": render_heatmap(change_map, after),
            "detections": render_detections(after, detections),
        }
    used_backends = sorted({candidate["matching_backend"] for registration in registrations
                            for candidate in registration.get("reference_candidates", [])})
    return {"coverage_percent": round(100 * total_scanned_area / total_reference_area, 1)
            if total_reference_area else 0.0,
            "reference_area_m2": round(total_reference_area, 3),
            "scanned_area_m2": round(total_scanned_area, 3),
            "matching_backend": " + ".join(used_backends) if used_backends else matcher.name,
            "registrations": registrations, "walls": walls}
