from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

from src.pipeline import ChangeDetectionPipeline


PROJECT_DIR = Path(__file__).parent
load_dotenv(PROJECT_DIR.parent / ".env")
load_dotenv(PROJECT_DIR / ".env")
CONFIG_PATH = PROJECT_DIR / "configs" / "default.yaml"


@st.cache_resource
def get_pipeline(dino_weight: float, ssim_weight: float, rgb_weight: float, threshold: float, min_area: int):
    pipeline = ChangeDetectionPipeline.from_yaml(CONFIG_PATH)
    pipeline.config["comparison"].update({"dino_weight": dino_weight, "ssim_weight": ssim_weight, "rgb_weight": rgb_weight})
    pipeline.config["detection"].update({"threshold": threshold, "min_area": min_area})
    return pipeline


st.set_page_config(page_title="Property Change Detection", layout="wide")
st.title("Comparaison d’états des lieux")
st.caption("Prototype V1 — détection de changements visuels entre une entrée et une sortie")

with st.sidebar:
    st.header("Images")
    before_file = st.file_uploader("Before", type=["jpg", "jpeg", "png", "webp"])
    after_file = st.file_uploader("After", type=["jpg", "jpeg", "png", "webp"])
    st.header("Paramètres")
    dino_weight = st.slider("Poids DINO", 0.0, 1.0, 0.6, 0.05)
    ssim_weight = st.slider("Poids SSIM", 0.0, 1.0, 0.3, 0.05)
    rgb_weight = st.slider("Poids RGB", 0.0, 1.0, 0.1, 0.05)
    threshold = st.slider("Seuil de changement", 0.0, 1.0, 0.4, 0.05)
    min_area = st.number_input("Surface minimale (pixels)", min_value=1, value=150, step=25)

if not before_file or not after_file:
    st.info("Chargez une image Before et une image After dans la barre latérale.")
    st.stop()

pipeline = get_pipeline(dino_weight, ssim_weight, rgb_weight, threshold, int(min_area))
with st.spinner("Analyse des images…"):
    result = pipeline.compare(before_file.getvalue(), after_file.getvalue(), PROJECT_DIR / "outputs")

report = result["report"]
st.success(f"{len(report['detected_changes'])} changement(s) détecté(s) — surface changée : {report['changed_surface_ratio']:.1%}")
st.caption(f"Backend features : {report['feature_backend']} | Matches : {report['alignment']['num_matches']} | Inliers : {report['alignment']['num_inliers']}")
if report.get("feature_backend_warning"):
    st.warning("Le modèle DINO n’est pas disponible ; le fallback local est utilisé. Vérifiez l’accès Hugging Face puis relancez l’application.")

views = [("Before", "before"), ("After", "after"), ("Before aligné", "aligned_before"), ("Correspondances", "matches"), ("Différence DINOv2", "dino_heatmap"), ("Différence fusionnée", "fused_heatmap"), ("Changements détectés", "detections")]
columns = st.columns(3)
for index, (title, key) in enumerate(views):
    with columns[index % 3]:
        st.subheader(title)
        st.image(result["visuals"][key], use_container_width=True)

st.subheader("Régions")
st.dataframe(report["regions"], use_container_width=True, hide_index=True)
st.subheader("Rapport JSON")
st.json(report)
