from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from .plan import FloorPlan


ROOM_COLORS = ("#dcebf9", "#e9f5e1", "#f9ebda", "#f0e5f6")


def _floor_mesh(room: Any, index: int) -> go.Mesh3d:
    x = [point[0] for point in room.polygon]
    y = [point[1] for point in room.polygon]
    z = [0.0] * len(room.polygon)
    # Fan triangulation is sufficient for the simple room polygons accepted by
    # the current JSON plan format.
    triangles = [(0, vertex, vertex + 1) for vertex in range(1, len(x) - 1)]
    return go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=[item[0] for item in triangles],
        j=[item[1] for item in triangles],
        k=[item[2] for item in triangles],
        color=ROOM_COLORS[index % len(ROOM_COLORS)],
        opacity=0.78,
        flatshading=True,
        name=room.label,
        hovertemplate=f"<b>{room.label}</b><extra></extra>",
        showlegend=False,
    )


def _wall_mesh(wall: Any) -> go.Mesh3d:
    x = [wall.start[0], wall.end[0], wall.end[0], wall.start[0]]
    y = [wall.start[1], wall.end[1], wall.end[1], wall.start[1]]
    z = [0.0, 0.0, wall.height, wall.height]
    return go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=[0, 0],
        j=[1, 2],
        k=[2, 3],
        color="#727b8c",
        customdata=[f"wall:{wall.id}"] * 4,
        meta=f"wall:{wall.id}",
        opacity=0.52,
        flatshading=True,
        name=wall.id,
        hovertemplate=(
            f"<b>{wall.id}</b><br>Pièce : {wall.room_id}"
            f"<br>Longueur : {wall.length:.2f} m<extra></extra>"
        ),
        showlegend=False,
    )


def build_interactive_figure(plan: FloorPlan, anomalies: list[dict[str, Any]] | None = None) -> go.Figure:
    """Build the freely navigable 3D plan and clickable anomaly markers."""
    figure = go.Figure()
    for index, room in enumerate(plan.rooms):
        figure.add_trace(_floor_mesh(room, index))
    for wall in plan.walls:
        figure.add_trace(_wall_mesh(wall))

    for room in plan.rooms:
        center_x = sum(point[0] for point in room.polygon) / len(room.polygon)
        center_y = sum(point[1] for point in room.polygon) / len(room.polygon)
        figure.add_trace(go.Scatter3d(
            x=[center_x], y=[center_y], z=[0.05], mode="text", text=[room.label],
            textfont={"color": "#374151", "size": 13}, hoverinfo="skip", showlegend=False,
        ))

    marker_x: list[float] = []
    marker_y: list[float] = []
    marker_z: list[float] = []
    marker_labels: list[str] = []
    marker_hover: list[str] = []
    marker_data: list[str] = []
    for index, anomaly in enumerate(anomalies or []):
        location = anomaly.get("location", {})
        try:
            wall = plan.wall(str(location.get("wall_id")))
        except KeyError:
            continue
        point = wall.point_at(float(location.get("u", 0.5)))
        marker_x.append(point[0])
        marker_y.append(point[1])
        marker_z.append(float(location.get("z_m", wall.height * 0.5)))
        anomaly_type = str(anomaly.get("type", "changement"))
        marker_labels.append(str(anomaly.get("id", index + 1)))
        marker_hover.append(
            f"<b>{anomaly_type}</b><br>Observation : {anomaly.get('observation_id', '—')}"
            f"<br>Mur : {wall.id}<br>Cliquez pour voir les images"
        )
        marker_data.append(f"anomaly:{index}")

    if marker_x:
        figure.add_trace(go.Scatter3d(
            x=marker_x,
            y=marker_y,
            z=marker_z,
            mode="markers+text",
            text=marker_labels,
            textposition="top center",
            customdata=marker_data,
            hovertext=marker_hover,
            hovertemplate="%{hovertext}<extra></extra>",
            marker={
                "size": 9,
                "color": "#e11d48",
                "line": {"color": "#fff", "width": 3},
                "symbol": "circle",
            },
            textfont={"color": "#9f1239", "size": 12},
            name="Différences",
            showlegend=False,
        ))

    figure.update_layout(
        template="plotly_white",
        margin={"l": 0, "r": 0, "t": 18, "b": 0},
        paper_bgcolor="#f8f9fc",
        scene={
            "dragmode": "orbit",
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.45, "y": -1.65, "z": 1.25}},
            "xaxis": {"title": "x (m)", "showbackground": False, "gridcolor": "#d9dde7"},
            "yaxis": {"title": "y (m)", "showbackground": False, "gridcolor": "#d9dde7"},
            "zaxis": {"title": "hauteur (m)", "showbackground": False, "gridcolor": "#d9dde7"},
        },
        hoverlabel={"bgcolor": "white", "font_size": 13, "font_family": "sans-serif"},
        uirevision=f"plan-{plan.id}",
    )
    return figure
