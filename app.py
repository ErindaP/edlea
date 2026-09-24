from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
import numpy as np
from PIL import Image
import streamlit as st
import streamlit.components.v2 as components_v2

from src.housing.interactive import build_interactive_figure
from src.housing.localization import localize_detections
from src.housing.plan import FloorPlan
from src.housing.multiview import ReferenceView, analyze_scan
from src.housing.store import HousingStore
from src.integrations.google_drive import GOOGLE_DRIVE_FOLDER_URL, download_drive_demo_pair
from src.pipeline import ChangeDetectionPipeline
from src.reporting.text import generate_text_report
from src.reporting.jobs import LocalReportJobs
from src.reporting.vision_language import DEFAULT_VLM_MODEL, LocalVisionReporter


PROJECT_DIR = Path(__file__).parent
load_dotenv(PROJECT_DIR.parent / ".env")
load_dotenv(PROJECT_DIR / ".env")
CONFIG_PATH = PROJECT_DIR / "configs" / "default.yaml"
HOUSING_ROOT = PROJECT_DIR / "data" / "housing"
PIPELINE_CACHE_VERSION = "pair-coverage-v2-plan-projection"


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
                 min_area: int, hysteresis_ratio: float, cache_version: str):
    # The explicit version prevents Streamlit hot reloads from returning an
    # instance created from an older ChangeDetectionPipeline implementation.
    del cache_version
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


def latest_scan_data(store: HousingStore, property_id: str, plan: FloorPlan) -> tuple[dict | None, dict, list[dict]]:
    latest = next((item for item in store.list_scans(property_id) if item.get("coverage")), None)
    if not latest:
        return None, {}, []
    coverage = latest["coverage"]
    masks = {}
    changes = []
    for wall_id, wall_data in coverage.get("walls", {}).items():
        status_path = Path(latest["directory"]) / wall_data["status_path"]
        if status_path.is_file():
            masks[wall_id] = np.asarray(Image.open(status_path).convert("L"))
        try:
            wall = plan.wall(wall_id)
        except KeyError:
            continue
        for change in wall_data.get("changes", []):
            changes.append({"id": change["id"], "type": "variation visuelle",
                            "confidence": change["score"], "observation_id": latest["id"],
                            "scan_id": latest["id"],
                            "evidence": ["Photos du scan : " + ", ".join(change.get("source_image_ids", []))]
                            if change.get("source_image_ids") else [],
                            "location": {"wall_id": wall_id, "room_id": wall.room_id,
                                         "u": change["u"], "z_m": wall.height * (1 - change["v"])}})
    return latest, masks, changes


def pair_coverage_data(store: HousingStore, property_id: str, plan: FloorPlan) -> tuple[dict[str, np.ndarray], dict | None]:
    """Load the latest persisted pairwise coverage layer for each selected wall."""
    masks: dict[str, np.ndarray] = {}
    latest = None
    for item in store.list_pair_coverages(property_id):
        try:
            plan.wall(item["wall_id"])
            status = np.asarray(Image.open(item["status_path"]).convert("L"))
        except (KeyError, OSError):
            continue
        masks[item["wall_id"]] = status
        latest = item
    return masks, latest


def execute_pair_comparison(
    before: bytes,
    after: bytes,
    *,
    property_id: str,
    observation_id: str,
    wall_id: str,
    plan: FloorPlan,
    properties: list[dict],
    source: str,
    use_local_vlm: bool,
    vlm_model_name: str,
    analyze_coverage: bool,
    pipeline_parameters: tuple[float, float, float, float, int, float],
) -> tuple[dict, Path]:
    """Persist and run one pair while keeping detector and LLM outputs independent."""
    pipeline = get_pipeline(*pipeline_parameters, PIPELINE_CACHE_VERSION)
    property_metadata = next(item for item in properties if item["id"] == property_id)
    metadata = {"property_id": property_id, "observation_id": observation_id, "wall_id": wall_id,
                "plan_id": plan.id, "source": source, "pair_coverage_enabled": analyze_coverage}
    observation_dir = store.add_observation(property_id, observation_id, before, after, metadata)
    result = pipeline.compare(before, after, observation_dir / "outputs", coverage_analysis=analyze_coverage)
    localized = localize_detections(result["detections"], plan, wall_id,
                                    result["source_images"]["after"].shape,
                                    result["report"]["detected_changes"])
    result["report"]["property"] = property_metadata
    result["report"]["observation_id"] = observation_id
    result["report"]["source"] = source
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
    return result, observation_dir


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
        ("Distance combinée", media.get("combined_distance")),
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
    coverage: dict[str, np.ndarray] | None = None,
) -> str | None:
    figure = build_interactive_figure(plan, anomalies, coverage)
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
            if anomaly.get("scan_id"):
                media = store.scan_change_media(property_id, observation_id, anomaly["location"]["wall_id"])
            else:
                media = store.observation_media(property_id, observation_id)
            show_anomaly_dialog(anomaly, media)

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
    if reset_message := st.session_state.pop("property_reset_message", None):
        st.success(reset_message)
    with st.expander("Réinitialiser le logement actif"):
        st.warning("Cette action supprime définitivement les comparaisons, rapports et scans multivues du logement actif. "
                   "Le logement et la géométrie de son plan sont conservés.")
        clear_references = st.checkbox(
            "Supprimer également les photos de référence calibrées",
            value=False,
            help="Laissez décoché pour pouvoir réutiliser les références lors du prochain scan.",
        )
        confirm_reset = st.checkbox("Je confirme la suppression des observations", value=False)
        if st.button("Vider le plan et réinitialiser les observations", disabled=not confirm_reset,
                     width="stretch"):
            try:
                removed = store.reset_property_data(selected_property, include_references=clear_references)
                for key in (
                    "last_observation_dir",
                    "last_result",
                    "last_property",
                    "last_anomaly_popup",
                    "selected_wall_id",
                    "comparison_success_message",
                ):
                    st.session_state.pop(key, None)
                details = f"{removed['observations']} comparaison(s) et {removed['scans']} scan(s) supprimés"
                if clear_references:
                    details += f", ainsi que {removed['references']} référence(s)"
                st.session_state["property_reset_message"] = details + ". Le plan est maintenant vierge."
                st.rerun()
            except (OSError, ValueError) as exc:
                st.error(f"La réinitialisation a échoué : {exc}")
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
latest_scan, scan_coverage_masks, scan_changes = latest_scan_data(store, selected_property, plan)
pair_coverage_masks, latest_pair_coverage = pair_coverage_data(store, selected_property, plan)
# A pairwise analysis is the most explicit, wall-specific action and therefore
# supersedes the latest multiview texture for the same wall.
coverage_masks = {**scan_coverage_masks, **pair_coverage_masks}
plan_tab, compare_tab, scan_tab, history_tab = st.tabs(["Plan 2.5D", "Nouvelle comparaison", "Scan multivue", "Historique"])

with plan_tab:
    st.subheader(plan.name)
    interactive_plan_3d(
        plan,
        anomalies + scan_changes,
        store,
        selected_property,
        key=f"overview-{selected_property}",
        height=720,
        coverage=coverage_masks,
    )
    st.caption("Rotation : clic gauche + déplacement · Zoom : molette · Déplacement : outil Pan de la barre. Cliquez sur un point rouge pour afficher les images de la comparaison.")
    st.caption("La localisation est une projection normalisée sur le mur choisi pour chaque paire d’images. Elle devient métrique après calibration de la prise de vue.")
    if latest_scan:
        st.caption(f"Dernier scan multivue : {latest_scan['name']} · Couverture des surfaces de référence : "
                   f"{latest_scan['coverage']['coverage_percent']:.1f} %. Vert : revu · Orange : non revu · Gris : non référencé.")
    if latest_pair_coverage:
        st.caption(
            f"Dernière couverture par paire : {latest_pair_coverage['observation_id']} sur "
            f"{latest_pair_coverage['wall_id']} · {latest_pair_coverage['coverage_of_before_percent']:.1f} % "
            "de la référence revue. Vert : revu · Orange : non revu."
        )
    st.dataframe([{"id": wall.id, "room_id": wall.room_id, "length_m": round(wall.length, 2), "height_m": wall.height} for wall in plan.walls], width="stretch", hide_index=True)

with compare_tab:
    st.subheader("Ajouter une observation")
    if comparison_message := st.session_state.pop("comparison_success_message", None):
        st.success(comparison_message)
    st.write("Déplacez le plan puis cliquez sur le mur concerné : il sera automatiquement sélectionné pour les photos ajoutées.")
    interactive_plan_3d(
        plan,
        anomalies,
        store,
        selected_property,
        key=f"plan-assignment-{selected_property}",
        height=600,
        allow_wall_selection=True,
        coverage=coverage_masks,
    )
    before_file = st.file_uploader("Image Before", type=["jpg", "jpeg", "png", "webp"], key="property_before")
    after_file = st.file_uploader("Image After", type=["jpg", "jpeg", "png", "webp"], key="property_after")
    observation_id = st.text_input("Identifiant de l’observation", value="inspection_sortie")
    wall_labels = {wall.id: f"{wall.id} — {wall.room_id}" for wall in plan.walls}
    selected_wall = st.session_state.get("selected_wall_id", plan.walls[0].id)
    wall_index = list(wall_labels).index(selected_wall) if selected_wall in wall_labels else 0
    wall_id = st.selectbox("Mur observé", list(wall_labels), index=wall_index, format_func=lambda key: wall_labels[key])
    st.session_state["selected_wall_id"] = wall_id
    analyze_coverage = st.toggle(
        "Analyser la couverture entre Avant et Après",
        value=False,
        help=("Utilise SuperPoint + LightGlue pour identifier la zone commune. Les parties de l’image Après "
              "sans correspondance fiable avec Avant sont exclues de la détection des différences."),
    )
    if analyze_coverage:
        st.caption("Mode conservateur activé : si le recouvrement ne peut pas être établi avec assez de certitude, "
                   "la comparaison est arrêtée au lieu d’interpréter les zones hors champ comme des défauts.")
    action_columns = st.columns(2)
    with action_columns[0]:
        analyze_upload = st.button("Analyser les images ajoutées", type="primary",
                                   disabled=not (before_file and after_file), width="stretch")
    with action_columns[1]:
        analyze_drive = st.button("Télécharger et analyser l’exemple Google Drive", width="stretch")
    st.caption(f"Le second bouton récupère `avant.jpg` et `apres.jpg` depuis le "
               f"[dossier Google Drive public]({GOOGLE_DRIVE_FOLDER_URL}), puis lance exactement la même pipeline et le rapport.")

    pair_to_analyze = None
    if analyze_upload:
        pair_to_analyze = (before_file.getvalue(), after_file.getvalue(), observation_id, "upload_utilisateur")
    elif analyze_drive:
        try:
            with st.spinner("Téléchargement et validation des deux images Google Drive…"):
                drive_before, drive_after = download_drive_demo_pair()
            pair_to_analyze = (drive_before, drive_after, f"drive_{observation_id}", GOOGLE_DRIVE_FOLDER_URL)
        except (RuntimeError, ValueError, OSError) as exc:
            st.error(f"Impossible de récupérer les images Google Drive : {exc}")

    if pair_to_analyze:
        before_bytes, after_bytes, effective_observation_id, source = pair_to_analyze
        try:
            with st.spinner("Alignement, comparaison et localisation sur le plan…"):
                result, observation_dir = execute_pair_comparison(
                    before_bytes,
                    after_bytes,
                    property_id=selected_property,
                    observation_id=effective_observation_id,
                    wall_id=wall_id,
                    plan=plan,
                    properties=properties,
                    source=source,
                    use_local_vlm=use_local_vlm,
                    vlm_model_name=vlm_model_name,
                    analyze_coverage=analyze_coverage,
                    pipeline_parameters=(dino_weight, ssim_weight, rgb_weight, threshold,
                                         int(min_area), hysteresis_ratio),
                )
            st.session_state["last_observation_dir"] = observation_dir
            st.session_state["last_result"] = result
            st.session_state["last_property"] = selected_property
            st.session_state["comparison_success_message"] = (
                f"Comparaison et rapport technique enregistrés dans {observation_dir.relative_to(PROJECT_DIR)}"
            )
            st.rerun()
        except (RuntimeError, ValueError, OSError) as exc:
            st.error(f"La comparaison n’a pas pu être exécutée : {exc}")

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

        pair_coverage = report.get("pair_coverage")
        if pair_coverage:
            st.subheader("Couverture géométrique Avant / Après")
            coverage_columns = st.columns(2)
            with coverage_columns[0]:
                st.metric("Part de l’image Avant retrouvée", f"{pair_coverage['coverage_of_before_percent']:.1f} %")
                st.metric("Part comparable dans l’image Après", f"{pair_coverage['comparable_after_percent']:.1f} %")
                st.caption(
                    f"Appariement : `{pair_coverage['matching_backend']}` · "
                    f"{pair_coverage['inliers']} correspondances validées sur {pair_coverage['matches']}."
                )
            with coverage_columns[1]:
                st.image(last_result["visuals"]["coverage_overlay"],
                         caption="Contour vert : zone analysée · Orange : zone Après exclue", width="stretch")

        refreshed_anomalies = all_anomalies(store, selected_property)
        interactive_plan_3d(
            plan,
            refreshed_anomalies,
            store,
            selected_property,
            key=f"result-{selected_property}",
            height=600,
            coverage=coverage_masks,
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

with scan_tab:
    st.subheader("Références et nouveau relevé multivue")
    st.caption("Prototype pour murs plans : des vues différentes et un nombre libre de photos sont acceptés si elles se recouvrent visuellement. "
               "La couverture est calculée uniquement sur les parties référencées des murs, pas sur les pièces entières.")
    st.info("Vous n’indiquez pas le mur des photos du nouveau scan : chaque photo est comparée à toutes les références. "
            "Le meilleur appariement géométrique choisit automatiquement la référence et donc le mur ; les candidats proches restent visibles ci-dessous.")
    demo_dir = PROJECT_DIR / "examples" / "multiview"
    demo_names = ("reference.jpg", "scan_gauche.jpg", "scan_marque.jpg")
    demo_ready = all((demo_dir / filename).is_file() for filename in demo_names)
    if st.button("Charger l’exemple multivue dans ce logement", disabled=not demo_ready,
                 help="Exécutez d’abord .venv/bin/python scripts/prepare_multiview_demo.py"):
        try:
            demo_wall = "living_east" if any(wall.id == "living_east" for wall in plan.walls) else plan.walls[0].id
            demo_reference = store.add_reference(selected_property, demo_wall, (demo_dir / "reference.jpg").read_bytes(),
                                                 (0.0, 0.0, 1.0, 1.0),
                                                 ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
                                                 "Exemple Wikimedia Commons")
            scan_dir, scan_records = store.add_scan(
                selected_property, "Exemple multivue", [(name, (demo_dir / name).read_bytes()) for name in demo_names[1:]],
            )
            reference_view = ReferenceView(demo_reference["id"], demo_wall,
                                           np.asarray(Image.open(demo_dir / "reference.jpg").convert("RGB")))
            scan_views = [(record["id"], np.asarray(Image.open(record["image_path"]).convert("RGB")))
                          for record in scan_records]
            with st.spinner("Analyse de l’exemple…"):
                store.save_scan_result(scan_dir, analyze_scan(plan, [reference_view], scan_views))
            st.rerun()
        except (ValueError, OSError) as exc:
            st.error(f"Impossible de charger l’exemple : {exc}")
    if demo_ready:
        st.caption("Source et licence : examples/multiview/SOURCE.txt. Les nouvelles vues sont simulées à partir de la photo source.")
    with st.expander("1. Ajouter une photo de référence", expanded=not store.list_references(selected_property)):
        with st.form(f"reference-form-{selected_property}"):
            reference_file = st.file_uploader("Photo de référence", type=["jpg", "jpeg", "png", "webp"], key="multiview_reference")
            reference_wall = st.selectbox("Mur représenté", [wall.id for wall in plan.walls], key="reference_wall")
            st.caption("Portion du mur couverte par les quatre coins indiqués dans la photo. "
                       "u va de gauche à droite ; v va du plafond au sol.")
            bounds_columns = st.columns(4)
            with bounds_columns[0]:
                u0 = st.number_input("u début", 0.0, 1.0, 0.0, 0.05)
            with bounds_columns[1]:
                v0 = st.number_input("v haut", 0.0, 1.0, 0.0, 0.05)
            with bounds_columns[2]:
                u1 = st.number_input("u fin", 0.0, 1.0, 1.0, 0.05)
            with bounds_columns[3]:
                v1 = st.number_input("v bas", 0.0, 1.0, 1.0, 0.05)
            corners_text = st.text_input("Coins du mur dans l’image (haut-gauche ; haut-droit ; bas-droit ; bas-gauche)",
                                         value="0,0;1,0;1,1;0,1",
                                         help="Coordonnées image normalisées entre 0 et 1. Le défaut suppose une photo recadrée sur le mur.")
            reference_submitted = st.form_submit_button("Enregistrer la référence")
            if reference_submitted:
                try:
                    if reference_file is None:
                        raise ValueError("Sélectionnez une photo.")
                    corners = tuple(tuple(float(value.strip()) for value in pair.split(","))
                                    for pair in corners_text.split(";"))
                    store.add_reference(selected_property, reference_wall, reference_file.getvalue(),
                                        (u0, v0, u1, v1), corners, reference_file.name)
                    st.success("Photo de référence enregistrée.")
                    st.rerun()
                except (ValueError, TypeError, OSError) as exc:
                    st.error(f"Calibration invalide : {exc}")
    references = store.list_references(selected_property)
    if references:
        st.dataframe([{"Photo": ref["source_name"] or ref["id"], "Mur": ref["wall_id"],
                       "Portion (u0,v0,u1,v1)": ref["bounds"], "ID": ref["id"]} for ref in references],
                     width="stretch", hide_index=True)
    else:
        st.info("Ajoutez une photo de référence pour démarrer. Recadrez-la sur un mur ou renseignez ses quatre coins.")

    st.markdown("#### 2. Nouveau scan")
    scan_name = st.text_input("Nom du relevé", value="nouvel_etat")
    scan_files = st.file_uploader("Photos du nouveau relevé (nombre libre)",
                                  type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True,
                                  key="multiview_scan")
    coverage_threshold = st.slider("Seuil des variations visuelles multivues", 0.1, 0.9, 0.32, 0.02)
    run_scan = st.button("Calculer la couverture et comparer", type="primary",
                         disabled=not (references and scan_files))
    if run_scan:
        try:
            reference_views = [ReferenceView(ref["id"], ref["wall_id"],
                                             np.asarray(Image.open(ref["image_path"]).convert("RGB")),
                                             tuple(ref["bounds"]), tuple(tuple(pair) for pair in ref["image_quad"]))
                               for ref in references]
            with st.spinner("Recalage des photos et calcul de couverture…"):
                scan_dir, scan_records = store.add_scan(selected_property, scan_name,
                                                        [(item.name, item.getvalue()) for item in scan_files])
                result = analyze_scan(plan, reference_views,
                                      [(record["id"], np.asarray(Image.open(record["image_path"]).convert("RGB")))
                                       for record in scan_records], threshold=coverage_threshold)
                store.save_scan_result(scan_dir, result)
            st.success(f"Scan enregistré : {scan_dir.relative_to(PROJECT_DIR)}")
            st.rerun()
        except (ValueError, OSError) as exc:
            st.error(f"Le scan n’a pas pu être analysé : {exc}")

    scans = [item for item in store.list_scans(selected_property) if item.get("coverage")]
    if scans:
        scan_ids = [item["id"] for item in scans]
        selected_scan_id = st.selectbox("Scan à consulter", scan_ids,
                                         format_func=lambda key: next(f"{item['name']} · {item['created_at'][:19]}"
                                                                      for item in scans if item["id"] == key))
        selected_scan = next(item for item in scans if item["id"] == selected_scan_id)
        summary = selected_scan["coverage"]
        st.metric("Couverture des surfaces de référence", f"{summary['coverage_percent']:.1f} %",
                  help="Somme des surfaces de mur revues divisée par la somme des surfaces de mur référencées.")
        st.caption(f"Surface référencée : {summary['reference_area_m2']:.2f} m² · "
                   f"Surface revue : {summary['scanned_area_m2']:.2f} m² · "
                   f"Appariement : `{summary.get('matching_backend', 'ancien scan / non renseigné')}`. "
                   "Les surfaces non référencées sont exclues.")
        registration_rows = summary.get("registrations", [])
        scan_names = {image["id"]: image.get("source_name", image["id"])
                      for image in selected_scan.get("images", [])}
        reference_names = {reference["id"]: reference.get("source_name") or reference["id"]
                           for reference in references}
        assignment_rows = []
        for registration in registration_rows:
            candidates = registration.get("reference_candidates", [])
            if not candidates:
                assignment_rows.append({"Photo du scan": scan_names.get(registration["image_id"], registration["image_id"]),
                                        "Statut": registration["status"], "Rang": "—",
                                        "Référence probable": "—", "Mur probable": "—",
                                        "Correspondances validées": 0, "Confiance": "—",
                                        "Moteur": "—", "Score relatif": "—"})
                continue
            for rank, candidate in enumerate(candidates, start=1):
                assignment_rows.append({
                    "Photo du scan": scan_names.get(registration["image_id"], registration["image_id"]),
                    "Statut": registration["status"], "Rang": rank,
                    "Référence probable": reference_names.get(candidate["reference_id"], candidate["reference_id"]),
                    "Mur probable": candidate["wall_id"],
                    "Correspondances validées": candidate["inliers"],
                    "Confiance": f"{candidate.get('mean_confidence', 0):.1%}",
                    "Moteur": candidate.get("matching_backend", "—"),
                    "Score relatif": f"{candidate['relative_score_percent']:.1f} %",
                })
        st.subheader("Attribution automatique des photos")
        st.dataframe(assignment_rows, width="stretch", hide_index=True)
        st.caption("Le rang 1 est retenu. Si une référence d’un autre mur obtient au moins 85 % de son score, "
                   "la photo est marquée `mur_ambigu` et n’augmente pas artificiellement la couverture.")
        if any(row["status"] != "localisee" for row in registration_rows):
            st.warning("Certaines photos n’ont pas été localisées avec assez de certitude ; elles ne sont pas comptées dans la couverture.")
        statuses = {}
        scan_markers = []
        for wall_key, wall_data in summary["walls"].items():
            path = Path(selected_scan["directory"]) / wall_data["status_path"]
            if path.is_file():
                statuses[wall_key] = np.asarray(Image.open(path).convert("L"))
            wall = plan.wall(wall_key)
            for change in wall_data.get("changes", []):
                scan_markers.append({"id": change["id"], "type": "variation visuelle", "confidence": change["score"],
                                     "observation_id": selected_scan_id, "scan_id": selected_scan_id,
                                     "evidence": ["Photos du scan : " + ", ".join(change.get("source_image_ids", []))]
                                     if change.get("source_image_ids") else [],
                                     "location": {"wall_id": wall_key, "room_id": wall.room_id,
                                                  "u": change["u"], "z_m": wall.height * (1 - change["v"])}})
        interactive_plan_3d(plan, scan_markers, store, selected_property,
                            key=f"scan-plan-{selected_property}-{selected_scan_id}", height=650, coverage=statuses)
        st.caption("Vert : revu · Orange : référence non revue · Gris : absence de référence. Cliquez sur un point rouge pour voir les comparaisons.")
        for wall_key, wall_data in summary["walls"].items():
            with st.expander(f"{wall_key} — {wall_data['coverage_percent']:.1f} % couverts · "
                             f"{len(wall_data['changes'])} variation(s)"):
                reference_labels = [reference_names.get(reference_id, reference_id)
                                    for reference_id in wall_data.get("reference_ids", [])]
                if reference_labels:
                    st.caption("Références utilisées sur ce mur : " + " · ".join(reference_labels))
                paths = store.scan_change_media(selected_property, selected_scan_id, wall_key)
                for column, (label, path) in zip(st.columns(3),
                                                 [("Référence rectifiée", paths["before"]),
                                                  ("Nouveau scan rectifié", paths["after"]),
                                                  ("Variations indicatives", paths["detections"])]):
                    with column:
                        if path.is_file():
                            st.image(path, caption=label, width="stretch")
                st.dataframe(wall_data["changes"], width="stretch", hide_index=True)

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
