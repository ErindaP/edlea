from __future__ import annotations

import json
from threading import Event

import numpy as np
from PIL import Image

from src.housing.store import HousingStore
from src.reporting.jobs import LocalReportJobs
from src.reporting.vision_language import build_visual_report_prompt


class BlockingReporter:
    model_name = "fake-local-vlm"

    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()

    def analyze(self, before, after, report):
        self.started.set()
        assert self.release.wait(timeout=5)
        return {"status": "success", "model": self.model_name, "text": "Une variation locale."}


def test_llm_job_leaves_technical_outputs_available_while_running(tmp_path):
    store = HousingStore(tmp_path / "housing")
    observation_dir = store.property_dir("maison") / "observations" / "visite"
    observation_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((12, 12, 3), dtype=np.uint8)).save(observation_dir / "before.jpg")
    Image.fromarray(np.zeros((12, 12, 3), dtype=np.uint8)).save(observation_dir / "after.jpg")
    (observation_dir / "outputs").mkdir()
    (observation_dir / "outputs" / "detections.png").write_bytes(b"technical image")
    report = {"run_id": "run-1", "detected_changes": []}
    store.save_report(observation_dir, report, "Rapport technique.")

    reporter = BlockingReporter()
    jobs = LocalReportJobs()
    assert jobs.submit(store, observation_dir, report, reporter)
    try:
        assert reporter.started.wait(timeout=5)
        assert jobs.status(observation_dir, "run-1") == "running"
        assert (observation_dir / "report.txt").read_text(encoding="utf-8") == "Rapport technique."
        assert (observation_dir / "outputs" / "detections.png").read_bytes() == b"technical image"
        assert store.load_llm_report(observation_dir, "run-1") is None
    finally:
        reporter.release.set()

    jobs._futures[(observation_dir.resolve(), "run-1")].result(timeout=5)
    assert store.load_llm_report(observation_dir, "run-1")["text"] == "Une variation locale."
    assert (observation_dir / "llm_report.txt").read_text(encoding="utf-8") == "Une variation locale."
    assert (observation_dir / "report.txt").read_text(encoding="utf-8") == "Rapport technique."


def test_superseded_llm_run_does_not_overwrite_new_comparison(tmp_path):
    store = HousingStore(tmp_path / "housing")
    observation_dir = store.property_dir("maison") / "observations" / "visite"
    observation_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((12, 12, 3), dtype=np.uint8)).save(observation_dir / "before.jpg")
    Image.fromarray(np.zeros((12, 12, 3), dtype=np.uint8)).save(observation_dir / "after.jpg")
    old_report = {"run_id": "old", "detected_changes": []}
    store.save_report(observation_dir, old_report, "old")

    reporter = BlockingReporter()
    jobs = LocalReportJobs()
    jobs.submit(store, observation_dir, old_report, reporter)
    assert reporter.started.wait(timeout=5)
    store.save_report(observation_dir, {"run_id": "new", "detected_changes": []}, "new")
    reporter.release.set()
    jobs._futures[(observation_dir.resolve(), "old")].result(timeout=5)

    assert store.load_llm_report(observation_dir, "new") is None


def test_visual_prompt_limits_bbox_context():
    report = {
        "changed_surface_ratio": 0.02,
        "detected_changes": [
            {"id": index, "type": "crack", "surface": "wall", "severity": "low",
             "confidence": 0.5, "bbox": [index, 2, 3, 4]}
            for index in range(20)
        ],
    }
    prompt = build_visual_report_prompt(report)
    context = json.loads(prompt.split(" :\n", maxsplit=1)[1].split("\nRéponds", maxsplit=1)[0])

    assert context["nombre_total_de_zones"] == 20
    assert context["types_et_effectifs"] == {"crack": 20}
    assert len(context["trois_exemples_prioritaires"]) == 3
    assert "bbox" not in prompt
