from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from src.housing.localization import localize_detections
from src.housing.plan import FloorPlan
from src.housing.render import build_plan_view, render_plan_25d, wall_at_pixel
from src.housing.store import HousingStore
from src.pipeline import ChangeDetectionPipeline
from src.reporting.text import generate_text_report


PROJECT_DIR = Path(__file__).parent
load_dotenv(PROJECT_DIR.parent / ".env")
load_dotenv(PROJECT_DIR / ".env")
CONFIG_PATH = PROJECT_DIR / "configs" / "default.yaml"
HOUSING_ROOT = PROJECT_DIR / "data" / "housing"


@st.cache_resource
def get_pipeline(dino_weight: float, ssim_weight: float, rgb_weight: float, threshold: float, min_area: int):
    pipeline = ChangeDetectionPipeline.from_yaml(CONFIG_PATH)
    pipeline.config["comparison"].update({"dino_weight": dino_weight, "ssim_weight": ssim_weight, "rgb_weight": rgb_weight})
    pipeline.config["detection"].update({"threshold": threshold, "min_area": min_area})
    return pipeline


def all_anomalies(store: HousingStore, property_id: str) -> list[dict]:
    anomalies = []
    for report in store.list_reports(property_id):
        anomalies.extend(item for item in report.get("detected_changes", []) if item.get("location"))
    return anomalies


def interactive_plan(plan: FloorPlan, anomalies: list[dict], key: str) -> str | None:
    image = render_plan_25d(plan, anomalies)
    click = streamlit_image_coordinates(image, width="content", key=key, cursor="pointer")
    if click and click.get("x") is not None and click.get("y") is not None:
        view = build_plan_view(plan, image.shape[1], image.shape[0])
        wall = wall_at_pixel(plan, view, click["x"], click["y"])
        if wall is not None:
            st.session_state["selected_wall_id"] = wall.id
            st.success(f"Mur sélectionné : {wall.id} ({wall.room_id})")
            return wall.id
        st.info("Le clic ne correspond pas à un mur. Cliquez sur une face grise du plan.")
    return st.session_state.get("selected_wall_id")


store = HousingStore(HOUSING_ROOT)
store.ensure_demo_property()
properties = store.list_properties()

st.set_page_config(page_title="Property Change Detection", layout="wide")
st.title("États des lieux géolocalisés")
st.caption("Détection de changements et projection des anomalies sur une représentation 2.5D du logement")

with st.sidebar:
    st.header("Logement")
    property_labels = {item["id"]: item["name"] for item in properties}
    selected_property = st.selectbox("Logement actif", list(property_labels), format_func=lambda key: property_labels[key])
    with st.expander("Ajouter un logement"):
        with st.form("new_property_form"):
            new_name = st.text_input("Nom du logement")
            custom_plan = st.file_uploader("Plan JSON optionnel", type=["json"], key="new_plan")
            submitted = st.form_submit_button("Créer le logement")
            if submitted and new_name.strip():
                try:
                    plan = FloorPlan.from_dict(json.loads(custom_plan.getvalue())) if custom_plan else FloorPlan.sample_house()
                    store.create_property(new_name.strip(), plan)
                    st.success("Logement créé. Rechargez la sélection si nécessaire.")
                    st.rerun()
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    st.error(f"Plan JSON invalide : {exc}")
    st.header("Paramètres de comparaison")
    dino_weight = st.slider("Poids DINO", 0.0, 1.0, 0.6, 0.05)
    ssim_weight = st.slider("Poids SSIM", 0.0, 1.0, 0.3, 0.05)
    rgb_weight = st.slider("Poids RGB", 0.0, 1.0, 0.1, 0.05)
    threshold = st.slider("Seuil de changement", 0.0, 1.0, 0.3, 0.05)
    min_area = st.number_input("Surface minimale (pixels)", min_value=1, value=100, step=25)

plan = store.load_plan(selected_property)
anomalies = all_anomalies(store, selected_property)
plan_tab, compare_tab, history_tab = st.tabs(["Plan 2.5D", "Nouvelle comparaison", "Historique"])

with plan_tab:
    st.subheader(plan.name)
    interactive_plan(plan, anomalies, key=f"plan_overview_{selected_property}")
    st.caption("La localisation est une projection normalisée sur le mur choisi pour chaque paire d’images. Elle devient métrique après calibration de la prise de vue.")
    st.dataframe([{"id": wall.id, "room_id": wall.room_id, "length_m": round(wall.length, 2), "height_m": wall.height} for wall in plan.walls], use_container_width=True, hide_index=True)

with compare_tab:
    st.subheader("Ajouter une observation")
    st.write("Cliquez sur le mur concerné ci-dessous : il sera automatiquement sélectionné pour les photos ajoutées.")
    interactive_plan(plan, anomalies, key=f"plan_assignment_{selected_property}")
    before_file = st.file_uploader("Image Before", type=["jpg", "jpeg", "png", "webp"], key="property_before")
    after_file = st.file_uploader("Image After", type=["jpg", "jpeg", "png", "webp"], key="property_after")
    observation_id = st.text_input("Identifiant de l’observation", value="inspection_sortie")
    wall_labels = {wall.id: f"{wall.id} — {wall.room_id}" for wall in plan.walls}
    selected_wall = st.session_state.get("selected_wall_id", plan.walls[0].id)
    wall_index = list(wall_labels).index(selected_wall) if selected_wall in wall_labels else 0
    wall_id = st.selectbox("Mur observé", list(wall_labels), index=wall_index, format_func=lambda key: wall_labels[key])
    st.session_state["selected_wall_id"] = wall_id
    analyze = st.button("Analyser et localiser les différences", type="primary", disabled=not (before_file and after_file))

    if analyze:
        pipeline = get_pipeline(dino_weight, ssim_weight, rgb_weight, threshold, int(min_area))
        property_metadata = next(item for item in properties if item["id"] == selected_property)
        metadata = {"property_id": selected_property, "observation_id": observation_id, "wall_id": wall_id, "plan_id": plan.id}
        observation_dir = store.add_observation(selected_property, observation_id, before_file.getvalue(), after_file.getvalue(), metadata)
        with st.spinner("Alignement, comparaison et localisation sur le plan…"):
            result = pipeline.compare(before_file.getvalue(), after_file.getvalue(), observation_dir / "outputs")
        localized = localize_detections(result["detections"], plan, wall_id, result["source_images"]["after"].shape, result["report"]["detected_changes"])
        result["report"]["property"] = property_metadata
        result["report"]["observation_id"] = observation_id
        result["report"]["plan"] = {"id": plan.id, "wall_id": wall_id}
        result["report"]["detected_changes"] = localized
        result["text_report"] = generate_text_report(result["report"])
        store.save_report(observation_dir, result["report"], result["text_report"])
        st.session_state["last_result"] = result
        st.session_state["last_property"] = selected_property
        st.success(f"Observation enregistrée dans {observation_dir.relative_to(PROJECT_DIR)}")

    last_result = st.session_state.get("last_result") if st.session_state.get("last_property") == selected_property else None
    if last_result:
        report = last_result["report"]
        st.image(render_plan_25d(plan, all_anomalies(store, selected_property)), use_container_width=True)
        st.success(f"{len(report['detected_changes'])} changement(s) localisé(s) sur {report['plan']['wall_id']}")
        st.text(last_result["text_report"])
        st.dataframe(report["detected_changes"], use_container_width=True, hide_index=True)
        columns = st.columns(3)
        for index, (title, key) in enumerate([("Before", "before"), ("After", "after"), ("Détections", "detections")]):
            with columns[index]:
                st.subheader(title)
                st.image(last_result["visuals"][key], use_container_width=True)

with history_tab:
    reports = store.list_reports(selected_property)
    if not reports:
        st.info("Aucune observation enregistrée pour ce logement.")
    for report in reports:
        with st.expander(report.get("observation_id", "Observation")):
            st.write(f"{len(report.get('detected_changes', []))} changement(s), seuil {report.get('detection_threshold', '?')}")
            st.dataframe(report.get("detected_changes", []), use_container_width=True, hide_index=True)
