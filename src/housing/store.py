from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image

from src.image_io import load_image

from .plan import FloorPlan


def _slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower()).strip("_")
    return value or "logement"


class HousingStore:
    """Filesystem-backed pseudo database for persistent demo properties."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def property_dir(self, property_id: str) -> Path:
        return self.root / property_id

    def list_properties(self) -> list[dict[str, Any]]:
        properties = []
        for directory in sorted(self.root.iterdir()):
            metadata_path = directory / "property.json"
            if directory.is_dir() and metadata_path.exists():
                properties.append(json.loads(metadata_path.read_text(encoding="utf-8")))
        return properties

    def create_property(self, name: str, plan: FloorPlan | None = None, property_id: str | None = None) -> dict[str, Any]:
        plan = plan or FloorPlan.sample_house()
        property_id = _slug(property_id or name)
        directory = self.property_dir(property_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "observations").mkdir(exist_ok=True)
        metadata = {"id": property_id, "name": name, "created_at": datetime.now(timezone.utc).isoformat(), "plan_id": plan.id}
        (directory / "property.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        (directory / "plan.json").write_text(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return metadata

    def ensure_demo_property(self) -> dict[str, Any]:
        existing = next((item for item in self.list_properties() if item["id"] == "demo_maison"), None)
        return existing or self.create_property("Maison de démonstration", FloorPlan.sample_house(), "demo_maison")

    def load_plan(self, property_id: str) -> FloorPlan:
        path = self.property_dir(property_id) / "plan.json"
        return FloorPlan.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def add_observation(self, property_id: str, observation_id: str, before: bytes, after: bytes,
                        metadata: dict[str, Any]) -> Path:
        directory = self.property_dir(property_id) / "observations" / _slug(observation_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "before.jpg").write_bytes(before)
        (directory / "after.jpg").write_bytes(after)
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        return directory

    def save_report(self, observation_dir: str | Path, report: dict[str, Any], text: str) -> None:
        directory = Path(observation_dir)
        (directory / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        (directory / "report.txt").write_text(text, encoding="utf-8")

    def save_llm_report(self, observation_dir: str | Path, analysis: dict[str, Any]) -> None:
        """Persist the independent LLM output without touching detector results."""
        directory = Path(observation_dir)
        (directory / "llm_report.json").write_text(
            json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        (directory / "llm_report.txt").write_text(analysis.get("text", ""), encoding="utf-8")

    def load_llm_report(self, observation_dir: str | Path, run_id: str | None = None) -> dict[str, Any] | None:
        path = Path(observation_dir) / "llm_report.json"
        if not path.is_file():
            return None
        try:
            analysis = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if run_id is not None and analysis.get("run_id") != run_id:
            return None
        return analysis

    def observation_media(self, property_id: str, observation_id: str) -> dict[str, Path]:
        """Return the persisted images associated with an observation."""
        directory = self.property_dir(property_id) / "observations" / _slug(observation_id)
        return {
            "before": directory / "before.jpg",
            "after": directory / "after.jpg",
            "detections": directory / "outputs" / "detections.png",
        }

    def list_reports(self, property_id: str) -> list[dict[str, Any]]:
        reports = []
        observations = self.property_dir(property_id) / "observations"
        if not observations.exists():
            return reports
        for report_path in sorted(observations.glob("*/report.json")):
            try:
                reports.append(json.loads(report_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
        return reports

    def add_reference(self, property_id: str, wall_id: str, image: bytes,
                      bounds: tuple[float, float, float, float],
                      image_quad: tuple[tuple[float, float], ...], source_name: str = "") -> dict[str, Any]:
        """Persist a calibrated reference photo without trusting uploaded filenames."""
        from .multiview import validate_reference

        self.load_plan(property_id).wall(wall_id)
        validate_reference(bounds, image_quad)
        decoded = load_image(image, 1280)
        reference_id = f"ref_{uuid4().hex[:12]}"
        directory = self.property_dir(property_id) / "references" / reference_id
        directory.mkdir(parents=True, exist_ok=False)
        Image.fromarray(decoded).save(directory / "image.jpg", quality=92)
        metadata = {"id": reference_id, "wall_id": wall_id, "bounds": list(bounds),
                    "image_quad": [list(point) for point in image_quad],
                    "source_name": Path(source_name).name, "created_at": datetime.now(timezone.utc).isoformat()}
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata

    def list_references(self, property_id: str) -> list[dict[str, Any]]:
        root = self.property_dir(property_id) / "references"
        if not root.exists():
            return []
        result = []
        for path in sorted(root.glob("*/metadata.json")):
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
                metadata["image_path"] = str(path.parent / "image.jpg")
                result.append(metadata)
            except (OSError, json.JSONDecodeError):
                continue
        return result

    def add_scan(self, property_id: str, name: str, images: list[tuple[str, bytes]]) -> tuple[Path, list[dict[str, Any]]]:
        if not images:
            raise ValueError("Ajoutez au moins une photo de scan.")
        decoded = [(filename, load_image(content, 1280)) for filename, content in images]
        scan_id = f"scan_{uuid4().hex[:12]}"
        directory = self.property_dir(property_id) / "scans" / scan_id
        image_dir = directory / "images"
        image_dir.mkdir(parents=True, exist_ok=False)
        records = []
        for index, (filename, image) in enumerate(decoded, start=1):
            image_id = f"photo_{index:03d}"
            Image.fromarray(image).save(image_dir / f"{image_id}.jpg", quality=92)
            records.append({"id": image_id, "source_name": Path(filename).name,
                            "image_path": str(image_dir / f"{image_id}.jpg")})
        metadata = {"id": scan_id, "name": name.strip() or scan_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "images": [{key: value for key, value in record.items() if key != "image_path"} for record in records]}
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        return directory, records

    def save_scan_result(self, directory: Path, result: dict[str, Any]) -> None:
        output_dir = directory / "outputs"
        output_dir.mkdir(exist_ok=True)
        walls = {}
        for wall_id, wall in result["walls"].items():
            safe_id = _slug(wall_id)
            for key in ("status", "reference_atlas", "scan_atlas", "change_heatmap", "detections"):
                Image.fromarray(wall[key]).save(output_dir / f"{safe_id}_{key}.png")
            walls[wall_id] = {key: value for key, value in wall.items()
                              if key not in {"status", "reference_atlas", "scan_atlas", "change_heatmap", "detections"}}
            walls[wall_id]["status_path"] = f"outputs/{safe_id}_status.png"
        summary = {**{key: value for key, value in result.items() if key != "walls"}, "walls": walls}
        (directory / "coverage.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    def list_scans(self, property_id: str) -> list[dict[str, Any]]:
        root = self.property_dir(property_id) / "scans"
        if not root.exists():
            return []
        result = []
        for path in sorted(root.glob("*/metadata.json"), reverse=True):
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
                coverage_path = path.parent / "coverage.json"
                if coverage_path.is_file():
                    metadata["coverage"] = json.loads(coverage_path.read_text(encoding="utf-8"))
                metadata["directory"] = str(path.parent)
                result.append(metadata)
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(result, key=lambda item: item.get("created_at", ""), reverse=True)

    def scan_change_media(self, property_id: str, scan_id: str, wall_id: str) -> dict[str, Path]:
        directory = self.property_dir(property_id) / "scans" / _slug(scan_id) / "outputs"
        prefix = _slug(wall_id)
        return {"before": directory / f"{prefix}_reference_atlas.png",
                "after": directory / f"{prefix}_scan_atlas.png",
                "detections": directory / f"{prefix}_detections.png"}
