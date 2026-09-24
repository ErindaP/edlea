import numpy as np
from PIL import Image

from src.housing.plan import FloorPlan
from src.housing.store import HousingStore


def test_observation_media_exposes_combined_distance(tmp_path):
    store = HousingStore(tmp_path / "housing")

    media = store.observation_media("maison", "visite 1")

    assert media["combined_distance"].name == "fused_heatmap.png"


def test_scan_media_exposes_combined_distance(tmp_path):
    store = HousingStore(tmp_path / "housing")

    media = store.scan_change_media("maison", "scan 1", "mur salon")

    assert media["combined_distance"].name == "mur_salon_change_heatmap.png"


def test_store_lists_persisted_pair_coverage_layer(tmp_path):
    store = HousingStore(tmp_path / "housing")
    store.create_property("Maison", FloorPlan.sample_house(), "maison")
    observation = store.property_dir("maison") / "observations" / "visite_1"
    outputs = observation / "outputs"
    outputs.mkdir(parents=True)
    Image.fromarray(np.array([[1, 2], [1, 2]], dtype=np.uint8)).save(outputs / "coverage_status.png")
    store.save_report(observation, {
        "observation_id": "visite_1",
        "plan": {"wall_id": "living_east"},
        "pair_coverage": {"coverage_of_before_percent": 50.0},
    }, "rapport")

    coverages = store.list_pair_coverages("maison")

    assert len(coverages) == 1
    assert coverages[0]["wall_id"] == "living_east"
    assert coverages[0]["status_path"] == outputs / "coverage_status.png"
