from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .plan import FloorPlan, Wall


ROOM_COLORS = [(224, 235, 249), (233, 245, 225), (249, 235, 218), (240, 229, 246)]


@dataclass(frozen=True)
class PlanView:
    bounds: tuple[float, float, float, float]
    scale: float
    origin: tuple[float, float]
    width: int
    height: int

    def project(self, point: tuple[float, float, float]) -> tuple[int, int]:
        x, y, z = point
        min_x, min_y, _, _ = self.bounds
        screen_x = self.origin[0] + ((x - min_x) - (y - min_y) * 0.55) * self.scale
        screen_y = self.origin[1] + ((x - min_x) * 0.28 + (y - min_y) * 0.65) * self.scale - z * self.scale * 0.55
        return int(round(screen_x)), int(round(screen_y))

    def wall_polygon(self, wall: Wall) -> np.ndarray:
        p1 = self.project((*wall.start, 0))
        p2 = self.project((*wall.end, 0))
        t1 = self.project((*wall.start, wall.height))
        t2 = self.project((*wall.end, wall.height))
        return np.array([p1, p2, t2, t1], dtype=np.int32)


def build_plan_view(plan: FloorPlan, width: int = 1100, height: int = 760) -> PlanView:
    points = [point for room in plan.rooms for point in room.polygon]
    min_x, min_y = min(p[0] for p in points), min(p[1] for p in points)
    max_x, max_y = max(p[0] for p in points), max(p[1] for p in points)
    bounds = (min_x, min_y, max_x, max_y)
    scale = min((width - 180) / max(max_x - min_x, 1), (height - 180) / max(max_y - min_y, 1))
    # First compute the projected extent, then translate it to the actual plot
    # center. This prevents plans with asymmetric coordinates from drifting.
    raw_view = PlanView(bounds, scale, (0.0, 0.0), width, height)
    projected = []
    for room in plan.rooms:
        projected.extend(raw_view.project((*point, 0)) for point in room.polygon)
    for wall in plan.walls:
        projected.extend(raw_view.wall_polygon(wall).tolist())
    min_px = min(point[0] for point in projected)
    max_px = max(point[0] for point in projected)
    min_py = min(point[1] for point in projected)
    max_py = max(point[1] for point in projected)
    target_center = (width / 2.0, (110.0 + height - 35.0) / 2.0)
    origin = (target_center[0] - (min_px + max_px) / 2.0,
              target_center[1] - (min_py + max_py) / 2.0)
    return PlanView(bounds, scale, origin, width, height)


def wall_at_pixel(plan: FloorPlan, view: PlanView, x: float, y: float, max_distance: float = 28.0) -> Wall | None:
    """Return the nearest wall face to a click in rendered-image pixels."""
    point = (float(x), float(y))
    candidates: list[tuple[float, Wall]] = []
    for wall in plan.walls:
        polygon = view.wall_polygon(wall)
        distance = float(cv2.pointPolygonTest(polygon, point, True))
        if distance >= 0:
            candidates.append((0.0, wall))
        elif abs(distance) <= max_distance:
            candidates.append((abs(distance), wall))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def render_plan_25d(plan: FloorPlan, anomalies: list[dict] | None = None,
                    width: int = 1100, height: int = 760) -> np.ndarray:
    view = build_plan_view(plan, width, height)
    canvas = np.full((height, width, 3), (248, 249, 252), dtype=np.uint8)

    for index, room in enumerate(plan.rooms):
        floor = np.array([view.project((x, y, 0)) for x, y in room.polygon], dtype=np.int32)
        cv2.fillPoly(canvas, [floor], ROOM_COLORS[index % len(ROOM_COLORS)])
        cv2.polylines(canvas, [floor], True, (160, 166, 178), 2, cv2.LINE_AA)

    for wall in plan.walls:
        polygon = view.wall_polygon(wall)
        cv2.fillPoly(canvas, [polygon], (113, 121, 137))
        cv2.polylines(canvas, [polygon], True, (56, 62, 76), 2, cv2.LINE_AA)

    # Labels are drawn last so the front wall faces cannot hide them.
    for room in plan.rooms:
        floor = np.array([view.project((x, y, 0)) for x, y in room.polygon], dtype=np.int32)
        center = np.mean(floor, axis=0).astype(int)
        cv2.putText(canvas, room.label, tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (55, 61, 74), 2, cv2.LINE_AA)

    cv2.putText(canvas, f"{plan.name} — représentation 2.5D", (35, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (34, 40, 52), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Cliquez sur un mur pour l’associer à une observation", (35, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (90, 96, 108), 1, cv2.LINE_AA)

    for anomaly in anomalies or []:
        location = anomaly.get("location", {})
        try:
            wall = plan.wall(location.get("wall_id"))
        except (KeyError, TypeError):
            continue
        point = wall.point_at(float(location.get("u", 0.5)))
        marker = view.project((point[0], point[1], float(location.get("z_m", wall.height * 0.5))))
        cv2.circle(canvas, marker, 11, (36, 45, 214), -1, cv2.LINE_AA)
        cv2.circle(canvas, marker, 15, (245, 245, 255), 2, cv2.LINE_AA)
        label = f"#{anomaly.get('id', '?')} {anomaly.get('type', 'change')}"
        cv2.putText(canvas, label, (marker[0] + 16, marker[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 45, 70), 2, cv2.LINE_AA)
    return canvas
