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


def _match(scan: np.ndarray, reference: ReferenceView, reference_mask: np.ndarray) -> dict | None:
    """Estimate scan->reference homography and reject weak/degenerate matches."""
    gray_scan = cv2.cvtColor(scan, cv2.COLOR_RGB2GRAY)
    gray_ref = cv2.cvtColor(reference.image, cv2.COLOR_RGB2GRAY)
    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=3500)
        norm = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(nfeatures=4000, fastThreshold=7)
        norm = cv2.NORM_HAMMING
    scan_keys, scan_desc = detector.detectAndCompute(gray_scan, None)
    ref_keys, ref_desc = detector.detectAndCompute(gray_ref, reference_mask)
    if scan_desc is None or ref_desc is None or len(scan_keys) < 8 or len(ref_keys) < 8:
        return None
    pairs = cv2.BFMatcher(norm).knnMatch(scan_desc, ref_desc, k=2)
    good = [first for pair in pairs if len(pair) == 2 for first, second in [pair]
            if first.distance < 0.72 * second.distance]
    if len(good) < 12:
        return None
    scan_points = np.float32([scan_keys[match.queryIdx].pt for match in good])
    ref_points = np.float32([ref_keys[match.trainIdx].pt for match in good])
    homography, inlier_flags = cv2.findHomography(scan_points, ref_points, cv2.RANSAC, 4.0)
    if homography is None or inlier_flags is None or not np.all(np.isfinite(homography)):
        return None
    inliers = inlier_flags.ravel().astype(bool)
    count = int(inliers.sum())
    if count < 10 or count / len(good) < 0.45:
        return None
    # Matching points on a small fixture alone cannot establish whole-wall coverage.
    points = scan_points[inliers]
    span = np.ptp(points, axis=0)
    reference_span = np.ptp(ref_points[inliers], axis=0)
    if (span[0] < scan.shape[1] * 0.12 or span[1] < scan.shape[0] * 0.12 or
            reference_span[0] < reference.image.shape[1] * 0.12 or
            reference_span[1] < reference.image.shape[0] * 0.12):
        return None
    corners = np.float32([[[0, 0], [scan.shape[1] - 1, 0],
                           [scan.shape[1] - 1, scan.shape[0] - 1], [0, scan.shape[0] - 1]]])
    projected = cv2.perspectiveTransform(corners, homography)[0]
    if (not np.all(np.isfinite(projected)) or not cv2.isContourConvex(projected) or
            cv2.contourArea(projected, oriented=True) <= 0):
        return None
    area = abs(cv2.contourArea(projected))
    ref_area = reference.image.shape[0] * reference.image.shape[1]
    if area < ref_area * 0.02 or area > ref_area * 20:
        return None
    return {"homography": homography, "inliers": count, "matches": len(good),
            "inlier_ratio": round(count / len(good), 3), "inlier_points": points}


def analyze_scan(plan: FloorPlan, references: list[ReferenceView], scans: list[tuple[str, np.ndarray]],
                 *, threshold: float = 0.32, min_area: int = 90) -> dict:
    """Compute per-wall coverage and indicative changes, independently of photo count."""
    if not references:
        raise ValueError("Ajoutez au moins une photo de référence calibrée.")
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
            match = _match(scan, view, source_mask)
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
                                    threshold=threshold, min_area=min_area, morph_kernel=3)
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
    return {"coverage_percent": round(100 * total_scanned_area / total_reference_area, 1)
            if total_reference_area else 0.0,
            "reference_area_m2": round(total_reference_area, 3),
            "scanned_area_m2": round(total_scanned_area, 3),
            "registrations": registrations, "walls": walls}
