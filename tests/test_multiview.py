from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from src.housing.interactive import build_interactive_figure
from src.housing.multiview import ReferenceView, analyze_scan, validate_reference
from src.housing.plan import FloorPlan
from src.housing.store import HousingStore


def textured_wall(seed: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(70, 210, (500, 640, 3), dtype=np.uint8)
    for index in range(35):
        center = (int(rng.integers(10, 630)), int(rng.integers(10, 490)))
        cv2.circle(image, center, 4 + index % 8, (25, 35, 45), 2)
    return image


def perspective_crop(image: np.ndarray, left: float, right: float) -> np.ndarray:
    height, width = image.shape[:2]
    src = np.float32([[left * width, 15], [right * width, 2],
                      [(right + 0.03) * width, height - 15], [(left + 0.03) * width, height - 30]])
    dst = np.float32([[0, 0], [479, 0], [479, 459], [0, 459]])
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(src, dst), (480, 460))


def test_multiple_views_cover_union_not_photo_count():
    image = textured_wall()
    reference = ReferenceView("r1", "living_east", image)
    left = perspective_crop(image, 0.0, 0.67)
    right = perspective_crop(image, 0.30, 0.95)
    plan = FloorPlan.sample_house()

    one = analyze_scan(plan, [reference], [("left", left)])
    both = analyze_scan(plan, [reference], [("left", left), ("right", right)])
    duplicate = analyze_scan(plan, [reference], [("left", left), ("right", right), ("again", left)])

    assert 45 < one["coverage_percent"] < 80
    assert one["coverage_percent"] < both["coverage_percent"] <= 100
    assert abs(both["coverage_percent"] - duplicate["coverage_percent"]) <= 0.2
    assert set(np.unique(both["walls"]["living_east"]["status"])) == {1, 2}


def test_unregistered_image_does_not_increase_coverage():
    image = textured_wall()
    result = analyze_scan(FloorPlan.sample_house(), [ReferenceView("r1", "living_east", image)],
                          [("blank", np.full((460, 480, 3), 127, np.uint8))])
    assert result["coverage_percent"] == 0
    assert result["registrations"][0]["status"] == "non_localisee"


def test_scan_auto_assigns_to_matching_wall_and_leaves_other_wall_uncovered():
    first = textured_wall(8)
    second = textured_wall(19)
    result = analyze_scan(FloorPlan.sample_house(),
                          [ReferenceView("r1", "living_east", first),
                           ReferenceView("r2", "bedroom_west", second)],
                          [("new", perspective_crop(second, 0.05, 0.90))])
    assert result["registrations"][0]["wall_id"] == "bedroom_west"
    assert result["registrations"][0]["reference_candidates"][0]["reference_id"] == "r2"
    assert result["registrations"][0]["reference_candidates"][0]["relative_score_percent"] == 100
    assert result["walls"]["living_east"]["coverage_percent"] == 0
    assert result["walls"]["bedroom_west"]["coverage_percent"] > 50


def test_changed_view_detects_mark_even_when_first_view_covers_it():
    image = textured_wall()
    changed = image.copy()
    cv2.line(changed, (340, 110), (365, 390), (0, 0, 0), 10)
    result = analyze_scan(FloorPlan.sample_house(), [ReferenceView("r1", "living_east", image)],
                          [("first", perspective_crop(image, 0.0, 0.70)),
                           ("marked", perspective_crop(changed, 0.25, 0.95))])
    changes = result["walls"]["living_east"]["changes"]
    assert changes
    assert any(0.5 < change["u"] < 0.65 for change in changes)
    assert any("marked" in change["source_image_ids"] for change in changes)


def test_calibration_rejects_degenerate_quad():
    try:
        validate_reference((0, 0, 1, 1), ((0, 0), (0.5, 0), (1, 0), (0.2, 0)))
    except ValueError:
        pass
    else:
        raise AssertionError("Degenerate quadrilateral should be rejected")


def test_store_roundtrip_and_plan_overlay(tmp_path):
    plan = FloorPlan.sample_house()
    store = HousingStore(tmp_path)
    store.create_property("Test", plan, "test")
    image = textured_wall()
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG")
    ref_record = store.add_reference("test", "living_east", buffer.getvalue(), (0, 0, 1, 1),
                                     ((0, 0), (1, 0), (1, 1), (0, 1)), "wall.jpg")
    scan_buffer = BytesIO()
    Image.fromarray(perspective_crop(image, 0.0, 0.67)).save(scan_buffer, format="JPEG")
    directory, records = store.add_scan("test", "premier relevé", [("left.jpg", scan_buffer.getvalue())])
    reference = ReferenceView(ref_record["id"], "living_east", image)
    result = analyze_scan(plan, [reference], [(records[0]["id"], perspective_crop(image, 0.0, 0.67))])
    store.save_scan_result(directory, result)
    scans = store.list_scans("test")

    assert len(store.list_references("test")) == 1
    assert scans[0]["coverage"]["coverage_percent"] > 0
    status = result["walls"]["living_east"]["status"]
    figure = build_interactive_figure(plan, coverage={"living_east": status})
    assert any(trace.name == "Référence non revue" for trace in figure.data)
    assert any(trace.name == "Revue dans le nouveau scan" for trace in figure.data)
