# Prototype de comparaison d’états des lieux

Cette première démo compare deux photos d’une même zone : alignement 2D, features denses, métriques pixel/SSIM, fusion des cartes, régions, détections et rapport JSON.

## Installation

Python 3.12 est requis. L’environnement isolé du projet est `.venv` :

```bash
cd /home/yanis/property_change_detection
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Le fichier `.env` parent (`/home/yanis/.env`) peut contenir `HF_TOKEN`. Il est chargé automatiquement par l’application et n’est jamais écrit dans le rapport.

## Lancer la démo

```bash
.venv/bin/streamlit run app.py
```

Chargez `Before` et `After`, puis ajustez les poids et le seuil dans la barre latérale. Les artefacts sont écrits dans `outputs/` : `aligned_before.png`, `matches.png`, `dino_heatmap.png`, `fused_heatmap.png`, `detections.png` et `report.json`.

## Backend de features

L’application utilise par défaut DINOv2 `facebook/dinov2-base` via Hugging Face avec `HF_TOKEN`. Ce modèle ViT-B/14 (86M paramètres) est adapté à la RTX 5070 Ti Laptop 12 Go, avec une résolution d’entrée de 518 px et une inférence à batch 1. Si le modèle n’est pas accessible, elle utilise un descripteur dense local afin de garder la pipeline fonctionnelle ; le rapport indique le backend utilisé. DINOv3 pourra remplacer DINOv2 après validation de l’accès au dépôt gated.

La segmentation V1 est une abstraction heuristique (`Segmenter`) produisant `wall` et `floor`, prête à être remplacée par SAM. L’alignement est une abstraction `RegistrationBackend`, avec ORB + homographie comme backend local et point d’extension pour RoMa, LoFTR ou RGB-D.

## Tests rapides

```bash
.venv/bin/python -m pytest -q
```
