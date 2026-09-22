# Prototype de comparaison d’états des lieux

Cette démo compare deux photos d’une même zone : alignement 2D, features denses, métriques pixel/SSIM, fusion des cartes, régions, classification indicative des anomalies et rapport JSON/textuel.

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

Chargez `Before` et `After`, puis ajustez les poids et le seuil dans la barre latérale. Les artefacts sont écrits dans `outputs/` : `aligned_before.png`, `matches.png`, `dino_heatmap.png`, `fused_heatmap.png`, `detections.png`, `report.json`, `report.txt` et les crops dans `anomalies/`.

## Backend de features

L’application utilise par défaut DINOv2 `facebook/dinov2-base` via Hugging Face avec `HF_TOKEN`. Ce modèle ViT-B/14 (86M paramètres) est adapté à la RTX 5070 Ti Laptop 12 Go, avec une résolution d’entrée de 518 px et une inférence à batch 1. Si le modèle n’est pas accessible, elle utilise un descripteur dense local afin de garder la pipeline fonctionnelle ; le rapport indique le backend utilisé. DINOv3 pourra remplacer DINOv2 après validation de l’accès au dépôt gated.

La segmentation V1 est une abstraction heuristique (`Segmenter`) produisant `wall` et `floor`, prête à être remplacée par SAM. L’alignement est une abstraction `RegistrationBackend`, avec ORB + homographie comme backend local et point d’extension pour RoMa, LoFTR ou RGB-D.

## Classification et rapport

La branche `rapport` ajoute une classification explicable après la détection. Pour chaque bounding box, le système calcule la forme, l’élongation, le remplissage, la compacité, la texture, les contours et l’intensité du changement. Des règles prudentes proposent `crack`, `impact`, `stain_or_dirt`, `object_change`, `paint_peeling` ou `unknown`, avec une confiance, une sévérité indicative et des éléments de preuve. Les zones ambiguës restent `unknown`.

La comparaison et ses sorties (`outputs/detections.png`, cartes de distance/changement, `report.json`, `report.txt`) sont enregistrées et affichées dès la fin de la détection. Ensuite seulement, une tâche locale facultative compare les images Avant/Après avec `Qwen/Qwen3-VL-2B-Instruct`. Elle s’exécute en arrière-plan et écrit **séparément** `llm_report.json` et `llm_report.txt`. L’interface actualise son statut pendant que les images restent consultables. Ce VLM occupe environ 4,3 Go sur disque, fonctionne sur la RTX 5070 Ti 12 Go et n’envoie pas les photos vers une API. Son premier lancement télécharge les poids dans le cache Hugging Face. Il peut être désactivé ou remplacé dans la barre latérale.

Le contexte VLM contient le nombre total de zones, un décompte par type et au plus trois exemples prioritaires ; il ne liste pas toutes les bounding boxes. La sortie brute est conservée dans `llm_report.json` (`raw_text`) pour audit. Le texte présenté passe par un garde-fou : les affirmations non observables sur une photo 2D, comme la profondeur ou la solidité, sont retirées. Pour une surface modifiée très faible, la synthèse parle d’une variation locale à confirmer au lieu d’affirmer une aggravation globale.

Le rapport texte reste une synthèse visuelle et non une expertise : il ne conclut ni à la responsabilité, ni au coût, ni à la nature certaine d’un dommage.

Pour les fissures et rayures, l’ouverture morphologique est désactivée par défaut : une ouverture carrée `5x5` risquerait de supprimer une ligne fine. Le seuil fort par défaut est `0.3`. Une hystérésis à `0.35 × seuil` étend ensuite chaque graine fiable aux portions plus faibles mais connectées. Cela produit des boîtes couvrant mieux les défauts fins sans transformer tout le bruit faible en détection. Les deux paramètres restent réglables dans l’interface. Le rapport indique aussi le score maximal observé lorsqu’aucune composante ne passe les filtres.

## Logements et plan 2.5D

La démo contient une pseudo-base de données sur disque dans `data/housing/`. Chaque logement est organisé ainsi :

```text
data/housing/<logement>/
├── property.json
├── plan.json
└── observations/<inspection>/
    ├── before.jpg
    ├── after.jpg
    ├── metadata.json
    ├── report.json
    ├── report.txt
    ├── llm_report.json (si l’analyse locale est activée)
    ├── llm_report.txt  (si l’analyse locale est activée)
    ├── outputs/
    └── anomalies/
```

Le logement de démonstration contient un plan synthétique de quatre pièces et des murs extrudés à `2.6 m`, rendus sous forme de représentation 2.5D. Dans l’onglet `Nouvelle comparaison`, chaque paire d’images est associée à un mur. Le centre de chaque anomalie est converti en coordonnées normalisées `(u, v)` sur ce mur, puis en coordonnées métriques approximatives `(x_m, y_m, z_m)`. Les marqueurs sont ensuite affichés sur le plan et conservés dans le rapport JSON.

Cette localisation est une première approximation : elle suppose que l’image couvre principalement le mur sélectionné. Une calibration par points correspondants ou une estimation de pose caméra sera nécessaire pour obtenir une localisation métrique précise.

Dans l’onglet `Plan 2.5D` ou `Nouvelle comparaison`, le plan peut être pivoté, déplacé et zoomé. Un clic sur une face de mur sélectionne automatiquement son identifiant ; les images ajoutées dans `Nouvelle comparaison` sont alors associées à ce mur. Un clic sur un marqueur rouge ouvre les images Avant, Après et Détections de la comparaison correspondante.

## Tests rapides

```bash
.venv/bin/python -m pytest -q
```
