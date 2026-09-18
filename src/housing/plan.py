from __future__ import annotations

from dataclasses import dataclass
from typing import Any


Point2D = tuple[float, float]


@dataclass(frozen=True)
class Wall:
    id: str
    room_id: str
    start: Point2D
    end: Point2D
    height: float = 2.6
    thickness: float = 0.12

    @property
    def length(self) -> float:
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return (dx * dx + dy * dy) ** 0.5

    def point_at(self, u: float) -> Point2D:
        u = max(0.0, min(1.0, float(u)))
        return (self.start[0] + u * (self.end[0] - self.start[0]),
                self.start[1] + u * (self.end[1] - self.start[1]))


@dataclass(frozen=True)
class Room:
    id: str
    label: str
    polygon: tuple[Point2D, ...]


@dataclass(frozen=True)
class FloorPlan:
    id: str
    name: str
    units: str
    rooms: tuple[Room, ...]
    walls: tuple[Wall, ...]

    def wall(self, wall_id: str) -> Wall:
        for wall in self.walls:
            if wall.id == wall_id:
                return wall
        raise KeyError(f"Unknown wall: {wall_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "units": self.units,
            "rooms": [{"id": room.id, "label": room.label, "polygon": [list(point) for point in room.polygon]} for room in self.rooms],
            "walls": [{"id": wall.id, "room_id": wall.room_id, "start": list(wall.start), "end": list(wall.end),
                       "height": wall.height, "thickness": wall.thickness} for wall in self.walls],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FloorPlan":
        rooms = tuple(Room(str(item["id"]), str(item["label"]), tuple((float(point[0]), float(point[1])) for point in item["polygon"])) for item in data.get("rooms", []))
        walls = tuple(Wall(str(item["id"]), str(item["room_id"]), tuple(item["start"]), tuple(item["end"]),
                          float(item.get("height", 2.6)), float(item.get("thickness", 0.12))) for item in data.get("walls", []))
        return cls(str(data["id"]), str(data["name"]), str(data.get("units", "m")), rooms, walls)

    @classmethod
    def sample_house(cls) -> "FloorPlan":
        """Small reproducible plan used by the demo and tests."""
        rooms = (
            Room("living", "Salon", ((0, 0), (5, 0), (5, 4), (0, 4))),
            Room("kitchen", "Cuisine", ((5, 0), (8, 0), (8, 4), (5, 4))),
            Room("bedroom", "Chambre", ((0, 4), (4, 4), (4, 7), (0, 7))),
            Room("bathroom", "Salle de bain", ((4, 4), (8, 4), (8, 7), (4, 7))),
        )
        walls = (
            Wall("living_north", "living", (0, 0), (5, 0)), Wall("living_east", "living", (5, 0), (5, 4)),
            Wall("living_south", "living", (5, 4), (0, 4)), Wall("living_west", "living", (0, 4), (0, 0)),
            Wall("kitchen_north", "kitchen", (5, 0), (8, 0)), Wall("kitchen_east", "kitchen", (8, 0), (8, 4)),
            Wall("kitchen_south", "kitchen", (8, 4), (5, 4)),
            Wall("bedroom_north", "bedroom", (0, 4), (4, 4)), Wall("bedroom_west", "bedroom", (0, 7), (0, 4)),
            Wall("bedroom_south", "bedroom", (4, 7), (0, 7)),
            Wall("bathroom_north", "bathroom", (4, 4), (8, 4)), Wall("bathroom_east", "bathroom", (8, 7), (8, 4)),
            Wall("bathroom_south", "bathroom", (8, 7), (4, 7)),
        )
        return cls("sample_house", "Maison de démonstration", "m", rooms, walls)

