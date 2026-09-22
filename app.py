from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v2 as components_v2

from src.housing.interactive import build_interactive_figure
from src.housing.localization import localize_detections
from src.housing.plan import FloorPlan
from src.housing.store import HousingStore
from src.pipeline import ChangeDetectionPipeline
from src.reporting.text import generate_text_report
from src.reporting.jobs import LocalReportJobs
from src.reporting.vision_language import DEFAULT_VLM_MODEL, LocalVisionReporter


PROJECT_DIR = Path(__file__).parent
load_dotenv(PROJECT_DIR.parent / ".env")
load_dotenv(PROJECT_DIR / ".env")
CONFIG_PATH = PROJECT_DIR / "configs" / "default.yaml"
HOUSING_ROOT = PROJECT_DIR / "data" / "housing"


PLOTLY_CLICK_BRIDGE_JS = r"""
export default function(component) {
    const { data, setTriggerValue } = component;
    let plot = null;
    let clickHandler = null;

    const detach = () => {
        if (plot && clickHandler && typeof plot.removeListener === "function") {
            plot.removeListener("plotly_click", clickHandler);
        }
        plot = null;
        clickHandler = null;
    };

    const bind = () => {
        const chartRoot = document.querySelector(`.st-key-${CSS.escape(data.chartKey)}`);
        const nextPlot = chartRoot && chartRoot.querySelector(".js-plotly-plot");
        if (!nextPlot || nextPlot === plot || typeof nextPlot.on !== "function") return;
        detach();
        plot = nextPlot;
        clickHandler = event => {
            const point = event && event.points && event.points[0];
            if (!point) return;
            let customdata = point.customdata;
            if (customdata === undefined && point.data) customdata = point.data.meta;
            if (customdata === undefined || customdata === null) return;
            setTriggerValue("clicked", {
                customdata,
                curveNumber: point.curveNumber,
                pointNumber: point.pointNumber,
                nonce: `${Date.now()}-${Math.random()}`,
            });
        };
        plot.on("plotly_click", clickHandler);
    };

    bind();
    const observer = new MutationObserver(bind);
    observer.observe(document.body, { childList: true, subtree: true });
    return () => {
        observer.disconnect();
        detach();
    };
}
"""


plotly_click_bridge = components_v2.component(
    "plotly_click_bridge",
    js=PLOTLY_CLICK_BRIDGE_JS,
)


@st.cache_resource
def get_local_vision_reporter(model_name: str) -> LocalVisionReporter:
    return LocalVisionReporter(model_name)


@st.cache_resource
def get_report_jobs() -> LocalReportJobs:
    return LocalReportJobs()


@st.cache_resource
def get_pipeline(dino_weight: float, ssim_weight: float, rgb_weight: float, threshold: float,
                 min_area: int, hysteresis_ratio: float):
    pipeline = ChangeDetectionPipeline.from_yaml(CONFIG_PATH)
    pipeline.config["comparison"].update({"dino_weight": dino_weight, "ssim_weight": ssim_weight, "rgb_weight": rgb_weight})
    pipeline.config["detection"].update({"threshold": threshold, "min_area": min_area,
                                         "hysteresis_ratio": hysteresis_ratio})
    return pipeline


@st.fragment(run_every="2s")
def show_llm_report(observation_dir: Path, run_id: str, observation_id: str) -> None:
    analysis = store.load_llm_report(observation_dir, run_id)
    job_status = get_report_jobs().status(observation_dir, run_id)
    st.subheader("Synthèse visuelle LLM")
    if job_status == "running":
        st.info("Le modèle local rédige la synthèse. Les images, boîtes et mesures ci-dessus sont déjà disponibles.")
        return
    if analysis is None:
        if job_status == "error":
            st.warning(f"La synthèse LLM n’a pas pu être enregistrée : {get_report_jobs().error(observation_dir, run_id)}")
        else:
            st.caption("Aucune synthèse LLM pour cette comparaison.")
        return
    if analysis.get("status") == "error":
        st.warning(f"Le modèle local a échoué : {analysis.get('error', 'erreur inconnue')}")
        return
    st.write(analysis.get("text", ""))
    st.caption(f"Modèle local : {analysis.get('model', '—')} · Résultat indicatif à vérifier sur les images.")
    st.download_button(
        "Télécharger la synthèse LLM",
        data=analysis.get("text", ""),
        file_name=f"rapport_llm_{observation_id}.txt",
        mime="text/plain",
        key=f"llm-download-{observation_dir}-{run_id}",
    )


def all_anomalies(store: HousingStore, property_id: str) -> list[dict]:
    anomalies = []
    for report in store.list_reports(property_id):
        observation_id = str(report.get("observation_id", ""))
        anomalies.extend(
            {**item, "observation_id": observation_id}
            for item in report.get("detected_changes", [])
            if item.get("location")
        )
    return anomalies


@st.dialog("Détail de la comparaison", width="large")
def show_anomaly_dialog(anomaly: dict, media: dict[str, Path]) -> None:
    location = anomaly.get("location", {})
    observation_id = str(anomaly.get("observation_id", "sans identifiant"))
    st.subheader(f"Différence #{anomaly.get('id', '?')} — {anomaly.get('type', 'changement')}")
    st.caption(f"Observation : {observation_id}")
    confidence = anomaly.get("confidence")
    confidence_label = f"{float(confidence):.0%}" if isinstance(confidence, (float, int)) else "—"
    st.write(
        f"**Mur :** {location.get('wall_id', '—')} · **Pièce :** {location.get('room_id', '—')} · "
        f"**Surface :** {anomaly.get('surface', '—')} · **Sévérité :** {anomaly.get('severity', '—')} · "
        f"**Confiance :** {confidence_label}"
    )

    available_images = [
        ("Avant", media.get("before")),
        ("Après", media.get("after")),
        ("Détections", media.get("detections")),
    ]
    available_images = [(label, path) for label, path in available_images if path and path.is_file()]
    if available_images:
        for column, (label, path) in zip(st.columns(len(available_images)), available_images):
            with column:
                st.image(path, caption=label, width="stretch")
    else:
        st.warning("Les images de cette observation ne sont plus disponibles.")

    evidence = anomaly.get("evidence", [])
    if evidence:
        st.caption("Indices : " + " · ".join(str(item) for item in evidence))


def interactive_plan_3d(
    plan: FloorPlan,
    anomalies: list[dict],
    store: HousingStore,
    property_id: str,
    *,
    key: str,
    height: int,
    allow_wall_selection: bool = False,
) -> str | None:
    figure = build_interactive_figure(plan, anomalies)
    figure.update_layout(height=height)
    st.plotly_chart(
        figure,
        key=key,
        width="stretch",
        theme=None,
        config={"scrollZoom": True, "displaylogo": False, "modeBarButtonsToRemove": ["toImage"]},
    )
    event = plotly_click_bridge(
        key=f"{key}-click-bridge",
        data={"chartKey": key},
        height=0,
        on_clicked_change=lambda: None,
    )
    clicked = getattr(event, "clicked", None)
    if not clicked:
        return st.session_state.get("selected_wall_id")
    customdata = clicked.get("customdata", "")
    if isinstance(customdata, (list, tuple)):
        customdata = customdata[0] if customdata else ""
    customdata = str(customdata)

    if customdata.startswith("anomaly:"):
        try:
            anomaly_index = int(customdata.split(":", maxsplit=1)[1])
            anomaly = anomalies[anomaly_index]
        except (ValueError, IndexError):
            return st.session_state.get("selected_wall_id")
        popup_token = f"{key}:{clicked.get('nonce', '')}"
        if st.session_state.get("last_anomaly_popup") != popup_token:
            st.session_state["last_anomaly_popup"] = popup_token
            observation_id = str(anomaly.get("observation_id", ""))
            show_anomaly_dialog(anomaly, store.observation_media(property_id, observation_id))

    if allow_wall_selection and customdata.startswith("wall:"):
        wall_id = customdata.split(":", maxsplit=1)[1]
        try:
            wall = plan.wall(wall_id)
        except KeyError:
            return st.session_state.get("selected_wall_id")
        st.session_state["selected_wall_id"] = wall.id
        st.success(f"Mur sélectionné : {wall.id} ({wall.room_id})")
        return wall.id

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
    hysteresis_ratio = st.slider(
        "Seuil d’extension des zones",
        0.10,
        1.00,
        0.35,
        0.05,
        help="Plus la valeur est basse, plus les boîtes s’étendent aux parties faibles mais connectées du changement.",
    )
    st.header("Génération du rapport")
    use_local_vlm = st.checkbox(
        "Analyse visuelle globale locale",
        value=True,
        help="Compare les images Avant/Après avec un petit modèle vision-langage exécuté sur cet ordinateur.",
    )
    vlm_model_name = st.text_input("Modèle vision-langage", value=DEFAULT_VLM_MODEL, disabled=not use_local_vlm)
    if use_local_vlm:
        st.caption("La comparaison s’affiche d’abord ; la synthèse locale arrive ensuite. Premier téléchargement du modèle recommandé : ~4,3 Go.")

plan = store.load_plan(selected_property)
anomalies = all_anomalies(store, selected_property)
plan_tab, compare_tab, history_tab = st.tabs(["Plan 2.5D", "Nouvelle comparaison", "Historique"])

with plan_tab:
    st.subheader(plan.name)
    interactive_plan_3d(
        plan,
        anomalies,
        store,
        selected_property,
        key=f"overview-{selected_property}",
        height=720,
    )
    st.caption("Rotation : clic gauche + déplacement · Zoom : molette · Déplacement : outil Pan de la barre. Cliquez sur un point rouge pour afficher les images de la comparaison.")
    st.caption("La localisation est une projection normalisée sur le mur choisi pour chaque paire d’images. Elle devient métrique après calibration de la prise de vue.")
    st.dataframe([{"id": wall.id, "room_id": wall.room_id, "length_m": round(wall.length, 2), "height_m": wall.height} for wall in plan.walls], width="stretch", hide_index=True)

with compare_tab:
    st.subheader("Ajouter une observation")
    st.write("Déplacez le plan puis cliquez sur le mur concerné : il sera automatiquement sélectionné pour les photos ajoutées.")
    interactive_plan_3d(
        plan,
        anomalies,
        store,
        selected_property,
        key=f"plan-assignment-{selected_property}",
        height=600,
        allow_wall_selection=True,
    )
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
        pipeline = get_pipeline(
            dino_weight,
            ssim_weight,
            rgb_weight,
            threshold,
            int(min_area),
            hysteresis_ratio,
        )
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
        result["report"]["run_id"] = uuid4().hex
        result["text_report"] = generate_text_report(result["report"])
        store.save_report(observation_dir, result["report"], result["text_report"])
        if use_local_vlm:
            get_report_jobs().submit(
                store,
                observation_dir,
                result["report"],
                get_local_vision_reporter(vlm_model_name.strip() or DEFAULT_VLM_MODEL),
            )
        st.session_state["last_observation_dir"] = observation_dir
        st.session_state["last_result"] = result
        st.session_state["last_property"] = selected_property
        st.success(f"Comparaison et rapport technique enregistrés dans {observation_dir.relative_to(PROJECT_DIR)}")

    last_result = st.session_state.get("last_result") if st.session_state.get("last_property") == selected_property else None
    if last_result:
        report = last_result["report"]
        st.subheader("Images de comparaison — disponibles immédiatement")
        columns = st.columns(3)
        for index, (title, image_key) in enumerate([("Avant", "before"), ("Après", "after"), ("Détections", "detections")]):
            with columns[index]:
                st.image(last_result["visuals"][image_key], caption=title, width="stretch")
        st.subheader("Cartes de changement")
        map_columns = st.columns(2)
        with map_columns[0]:
            st.image(last_result["visuals"]["dino_heatmap"], caption="Distance DINO", width="stretch")
        with map_columns[1]:
            st.image(last_result["visuals"]["fused_heatmap"], caption="Carte fusionnée", width="stretch")

        refreshed_anomalies = all_anomalies(store, selected_property)
        interactive_plan_3d(
            plan,
            refreshed_anomalies,
            store,
            selected_property,
            key=f"result-{selected_property}",
            height=600,
        )
        st.success(f"{len(report['detected_changes'])} changement(s) localisé(s) sur {report['plan']['wall_id']}")
        st.subheader("Rapport technique — détection et distances")
        st.text(last_result["text_report"])
        st.download_button(
            "Télécharger le rapport technique",
            data=last_result["text_report"],
            file_name=f"rapport_{report.get('observation_id', 'observation')}.txt",
            mime="text/plain",
            key="download_last_report",
        )
        st.dataframe(report["detected_changes"], width="stretch", hide_index=True)
        show_llm_report(
            st.session_state["last_observation_dir"],
            report["run_id"],
            str(report.get("observation_id", "observation")),
        )

with history_tab:
    reports = store.list_reports(selected_property)
    if not reports:
        st.info("Aucune observation enregistrée pour ce logement.")
    for report in reports:
        with st.expander(report.get("observation_id", "Observation")):
            st.write(f"{len(report.get('detected_changes', []))} changement(s), seuil {report.get('detection_threshold', '?')}")
            historical_text = generate_text_report(report)
            st.text(historical_text)
            st.download_button(
                "Télécharger ce rapport",
                data=historical_text,
                file_name=f"rapport_{report.get('observation_id', 'observation')}.txt",
                mime="text/plain",
                key=f"download_history_{selected_property}_{report.get('observation_id', 'observation')}",
            )
            st.dataframe(report.get("detected_changes", []), width="stretch", hide_index=True)
            if report.get("run_id"):
                history_dir = store.observation_media(
                    selected_property, str(report.get("observation_id", "")),
                )["before"].parent
                show_llm_report(history_dir, report["run_id"], str(report.get("observation_id", "observation")))
