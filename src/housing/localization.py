from __future__ import annotations

from typing import Any

from ..types import DetectedChange
from .plan import FloorPlan


def localize_detections(detections: list[DetectedChange], plan: FloorPlan, wall_id: str,
                        image_shape: tuple[int, int], annotations: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Map image-normalized detections onto the selected wall."""
    wall = plan.wall(wall_id)
    image_height, image_width = image_shape[:2]
    localized = []
    annotations_by_id = {int(item["id"]): item for item in (annotations or []) if "id" in item}
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        center_u = ((x1 + x2) / 2.0) / max(1, image_width)
        center_v = 1.0 - ((y1 + y2) / 2.0) / max(1, image_height)
        center_u = max(0.0, min(1.0, center_u))
        center_v = max(0.0, min(1.0, center_v))
        x, y = wall.point_at(center_u)
        localized.append({
            **annotations_by_id.get(detection.id, {}),
            **detection.to_dict(),
            "location": {
                "plan_id": plan.id,
                "wall_id": wall.id,
                "room_id": wall.room_id,
                "u": round(center_u, 6),
                "v": round(center_v, 6),
                "x_m": round(x, 4),
                "y_m": round(y, 4),
                "z_m": round(center_v * wall.height, 4),
                "mapping": "normalized_wall_projection",
            },
        })
    return localized
