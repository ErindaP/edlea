from __future__ import annotations

import cv2
import numpy as np

from .plan import FloorPlan, Point2D


ROOM_COLORS = [(224, 235, 249), (233, 245, 225), (249, 235, 218), (240, 229, 246)]


def _project(point: tuple[float, float, float], bounds: tuple[float, float, float, float], scale: float,
             origin: tuple[float, float]) -> tuple[int, int]:
    x, y, z = point
    min_x, min_y, _, _ = bounds
    screen_x = origin[0] + ((x - min_x) - (y - min_y) * 0.55) * scale
    screen_y = origin[1] + ((x - min_x) * 0.28 + (y - min_y) * 0.65) * scale - z * scale * 0.55
    return int(round(screen_x)), int(round(screen_y))


def render_plan_25d(plan: FloorPlan, anomalies: list[dict] | None = None,
                    width: int = 1100, height: int = 760) -> np.ndarray:
    points = [point for room in plan.rooms for point in room.polygon]
    min_x, min_y = min(p[0] for p in points), min(p[1] for p in points)
    max_x, max_y = max(p[0] for p in points), max(p[1] for p in points)
    bounds = (min_x, min_y, max_x, max_y)
    scale = min((width - 180) / max(max_x - min_x, 1), (height - 160) / max(max_y - min_y, 1))
    canvas = np.full((height, width, 3), (248, 249, 252), dtype=np.uint8)
    origin = (85.0, 145.0)

    for index, room in enumerate(plan.rooms):
        floor = np.array([_project((x, y, 0), bounds, scale, origin) for x, y in room.polygon], dtype=np.int32)
        cv2.fillPoly(canvas, [floor], ROOM_COLORS[index % len(ROOM_COLORS)])
        cv2.polylines(canvas, [floor], True, (160, 166, 178), 2, cv2.LINE_AA)
        center = np.mean(floor, axis=0).astype(int)
        cv2.putText(canvas, room.label, tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (70, 76, 88), 2, cv2.LINE_AA)

    # Draw the vertical faces of the walls to give the plan a lightweight 2.5D appearance.
    for wall in plan.walls:
        p1 = _project((*wall.start, 0), bounds, scale, origin)
        p2 = _project((*wall.end, 0), bounds, scale, origin)
        t1 = _project((*wall.start, wall.height), bounds, scale, origin)
        t2 = _project((*wall.end, wall.height), bounds, scale, origin)
        cv2.fillPoly(canvas, [np.array([p1, p2, t2, t1], dtype=np.int32)], (113, 121, 137))
        cv2.polylines(canvas, [np.array([p1, p2, t2, t1], dtype=np.int32)], True, (56, 62, 76), 2, cv2.LINE_AA)

    cv2.putText(canvas, f"{plan.name} — représentation 2.5D", (35, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (34, 40, 52), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Marqueurs rouges : différences localisées sur le mur associé", (35, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (90, 96, 108), 1, cv2.LINE_AA)

    for anomaly in anomalies or []:
        location = anomaly.get("location", {})
        wall_id = location.get("wall_id")
        try:
            wall = plan.wall(wall_id)
        except (KeyError, TypeError):
            continue
        point = wall.point_at(float(location.get("u", 0.5)))
        marker = _project((point[0], point[1], float(location.get("z_m", wall.height * 0.5))), bounds, scale, origin)
        cv2.circle(canvas, marker, 11, (36, 45, 214), -1, cv2.LINE_AA)
        cv2.circle(canvas, marker, 15, (245, 245, 255), 2, cv2.LINE_AA)
        label = f"#{anomaly.get('id', '?')} {anomaly.get('type', 'change')}"
        cv2.putText(canvas, label, (marker[0] + 16, marker[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 45, 70), 2, cv2.LINE_AA)
    return canvas

