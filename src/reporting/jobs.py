from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path
from threading import Lock
from typing import Any

from ..housing.store import HousingStore
from ..image_io import load_image
from .vision_language import LocalVisionReporter


class LocalReportJobs:
    """Run VLM reports off the Streamlit script thread, one GPU job at a time."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="local-report")
        self._futures: dict[tuple[Path, str], Future[dict[str, Any]]] = {}
        self._lock = Lock()

    def submit(
        self,
        store: HousingStore,
        observation_dir: Path,
        report: dict[str, Any],
        reporter: LocalVisionReporter,
    ) -> bool:
        observation_dir = observation_dir.resolve()
        run_id = str(report["run_id"])
        job_key = (observation_dir, run_id)
        with self._lock:
            existing = self._futures.get(job_key)
            if existing is not None and not existing.done():
                return False
            self._futures[job_key] = self._executor.submit(
                self._run, store, observation_dir, dict(report), reporter,
            )
        return True

    @staticmethod
    def _run(
        store: HousingStore,
        observation_dir: Path,
        report: dict[str, Any],
        reporter: LocalVisionReporter,
    ) -> dict[str, Any]:
        try:
            if not LocalReportJobs._is_current_run(observation_dir, str(report["run_id"])):
                return {"status": "superseded", "run_id": report["run_id"]}
            before = load_image(observation_dir / "before.jpg", max_size=1280)
            after = load_image(observation_dir / "after.jpg", max_size=1280)
            analysis = reporter.analyze(before, after, report)
        except Exception as exc:
            analysis = {
                "status": "error",
                "model": reporter.model_name,
                "text": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
        analysis["run_id"] = report["run_id"]
        if LocalReportJobs._is_current_run(observation_dir, str(report["run_id"])):
            store.save_llm_report(observation_dir, analysis)
        return analysis

    @staticmethod
    def _is_current_run(observation_dir: Path, run_id: str) -> bool:
        try:
            current = json.loads((observation_dir / "report.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return current.get("run_id") == run_id

    def status(self, observation_dir: Path, run_id: str) -> str:
        job_key = (observation_dir.resolve(), str(run_id))
        with self._lock:
            future = self._futures.get(job_key)
        if future is None:
            return "idle"
        if not future.done():
            return "running"
        return "error" if future.exception() is not None else "finished"

    def error(self, observation_dir: Path, run_id: str) -> str | None:
        job_key = (observation_dir.resolve(), str(run_id))
        with self._lock:
            future = self._futures.get(job_key)
        if future is None or not future.done():
            return None
        exception = future.exception()
        return f"{type(exception).__name__}: {exception}" if exception else None
