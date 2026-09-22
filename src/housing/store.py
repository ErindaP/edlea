from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
