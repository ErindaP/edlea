from src.housing.store import HousingStore


def test_observation_media_exposes_combined_distance(tmp_path):
    store = HousingStore(tmp_path / "housing")

    media = store.observation_media("maison", "visite 1")

    assert media["combined_distance"].name == "fused_heatmap.png"


def test_scan_media_exposes_combined_distance(tmp_path):
    store = HousingStore(tmp_path / "housing")

    media = store.scan_change_media("maison", "scan 1", "mur salon")

    assert media["combined_distance"].name == "mur_salon_change_heatmap.png"
