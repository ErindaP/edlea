from __future__ import annotations

from src.housing.plan import FloorPlan
from src.housing.store import HousingStore


def seed_property_data(store: HousingStore, property_id: str) -> None:
    base = store.property_dir(property_id)
    for name, child in (("observations", "inspection"), ("scans", "scan_1"), ("references", "ref_1")):
        directory = base / name / child
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "data.txt").write_text(name, encoding="utf-8")


def test_reset_clears_observations_and_scans_but_keeps_plan_and_references(tmp_path):
    store = HousingStore(tmp_path / "housing")
    store.create_property("Maison", FloorPlan.sample_house(), "maison")
    seed_property_data(store, "maison")

    removed = store.reset_property_data("maison")

    base = store.property_dir("maison")
    assert removed == {"observations": 1, "scans": 1}
    assert not list((base / "observations").iterdir())
    assert not list((base / "scans").iterdir())
    assert (base / "references" / "ref_1" / "data.txt").is_file()
    assert (base / "property.json").is_file()
    assert store.load_plan("maison").id == "sample_house"


def test_full_reset_also_clears_references(tmp_path):
    store = HousingStore(tmp_path / "housing")
    store.create_property("Maison", FloorPlan.sample_house(), "maison")
    seed_property_data(store, "maison")

    removed = store.reset_property_data("maison", include_references=True)

    assert removed == {"observations": 1, "scans": 1, "references": 1}
    assert not list((store.property_dir("maison") / "references").iterdir())


def test_reset_rejects_unknown_or_outside_property(tmp_path):
    store = HousingStore(tmp_path / "housing")
    store.create_property("Maison", FloorPlan.sample_house(), "maison")

    for invalid_id in ("inconnu", "../ailleurs"):
        try:
            store.reset_property_data(invalid_id)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Reset should reject {invalid_id}")
