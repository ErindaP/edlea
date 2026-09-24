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

Chargez `Before` et `After`, puis ajustez les poids et le seuil dans la barre latérale. Les artefacts sont écrits dans `outputs/` : `aligned_before.png`, `matches.png`, `dino_heatmap.png`, `fused_heatmap.png`, `detections.png`, `report.json`, `report.txt` et les crops dans `anomalies/`. Lorsque l’analyse de couverture par paire est activée, `coverage_overlay.png` montre également la zone comparable.

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

Dans l’onglet `Plan 2.5D` ou `Nouvelle comparaison`, le plan peut être pivoté, déplacé et zoomé. Un clic sur une face de mur sélectionne automatiquement son identifiant ; les images ajoutées dans `Nouvelle comparaison` sont alors associées à ce mur. Un clic sur un marqueur rouge ouvre les images Avant, Après, Détections et Distance combinée de la comparaison correspondante.

La barre latérale contient aussi `Réinitialiser le logement actif`. Après confirmation, cette action efface les comparaisons, rapports et scans multivues qui alimentent les marqueurs et la couverture du plan, sans supprimer le logement ni `plan.json`. Les références calibrées sont conservées par défaut ; une case séparée permet de les supprimer également pour repartir entièrement de zéro.

## Scan multivue et couverture (branche `multivue-couverture`)

L’onglet `Scan multivue` accepte plusieurs photos **de référence** par logement et un nombre libre de photos par nouveau relevé. Pour chaque référence, indiquez le mur, la portion du mur `(u0,v0,u1,v1)` et, si la photo n’est pas déjà recadrée sur ce mur, les quatre coins du mur dans l’image (coordonnées normalisées entre 0 et 1, dans l’ordre haut-gauche, haut-droit, bas-droit, bas-gauche). La valeur par défaut `0,0;1,0;1,1;0,1` suppose que *toute la photo est une seule face de mur* ; elle n’est pas adaptée à une vue de pièce montrant plusieurs murs, sol et plafond.

Les nouvelles photos ne demandent aucune indication de pièce ou de mur. Elles sont automatiquement rapprochées de **toutes** les références avec SuperPoint + LightGlue, puis une homographie est vérifiée par RANSAC. Le modèle officiel `ETH-CVG/lightglue_superpoint` est chargé via la dépendance `transformers` déjà installée et utilise le GPU s’il est disponible. S’il ne peut pas être chargé, si son inférence échoue ou s’il produit trop peu de correspondances, le traitement revient automatiquement à SIFT (puis ORB si nécessaire). L’interface classe jusqu’à trois références probables pour chaque photo avec leur mur, leur confiance et leur nombre de correspondances validées. Le premier résultat est retenu ; si deux murs obtiennent des scores trop proches, la photo est déclarée ambiguë et exclue du taux. L’application refuse aussi les correspondances faibles ou trop concentrées. Elle projette les zones reconnues sur une grille commune par mur : vert = vu dans le nouveau scan, orange = présent dans la référence mais non revu, gris = pas de référence. Les comparaisons de pixels/SSIM, leurs boîtes et images rectifiées sont indépendantes du rapport LLM et disponibles directement. Les points rouges du plan ouvrent les quatre images comparées, dont la carte de distance combinée. Chaque variation conserve également l’identifiant de la ou des photos du nouveau scan qui l’ont mise en évidence.

Attention à la licence : le code et les poids LightGlue sont sous Apache-2.0, mais les poids SuperPoint associés sont indiqués par leur fiche officielle comme réservés à la recherche académique/non lucrative. Ce backend convient donc à ce prototype expérimental ; il faudra le remplacer par un détecteur aux poids compatibles, par exemple ALIKED ou DISK + LightGlue, avant une exploitation commerciale.

Dans `Nouvelle comparaison`, le bouton **« Télécharger et analyser l’exemple Google Drive »** récupère `avant.jpg` et `apres.jpg` depuis le dossier public configuré, valide leur format, les enregistre dans le logement actif et lance la même détection que les fichiers importés. Le rapport technique est immédiatement disponible ; si l’analyse visuelle locale est activée, sa génération démarre ensuite en tâche de fond. Les identifiants des deux fichiers sont volontairement explicites dans `src/integrations/google_drive.py` : si les fichiers du dossier sont remplacés plutôt que mis à jour, ces identifiants doivent être adaptés.

Dans ce même onglet, l’option **« Analyser la couverture entre Avant et Après »** applique SuperPoint + LightGlue (avec le même repli local que le multivue) et vérifie l’homographie par RANSAC. Par défaut, l’image Avant entière est considérée comme une vue de la totalité du mur sélectionné. Lorsque les inliers sont nombreux et couvrent largement les deux images, l’emprise complète de l’image Après est projetée dans cette référence. Lorsque les correspondances sont localisées, plusieurs homographies de contrôle sont recalculées en retirant successivement chaque inlier : l’analyse conserve les zones où leurs projections restent cohérentes, même lorsqu’elles sont uniformes et dépourvues de points-clés. Une marge de sécurité est retirée et les composantes raccordées à la frontière fiable sont ignorées. L’orientation EXIF des JPEG est appliquée avant le calcul afin que les axes de l’image correspondent à ceux du mur. Le rapport indique la part de l’image Avant retrouvée dans Après et la part de l’image Après réellement comparable. Les distances, détections et bounding boxes sont calculées uniquement dans ce masque ; les zones orange de `coverage_overlay.png` sont donc ignorées. Le masque est aussi projeté sur le mur sélectionné dans les plans 2.5D : vert pour la partie revue dans Après, orange pour la partie de la référence Avant non revue. Si le recouvrement n’est pas assez fiable, l’analyse s’arrête au lieu de produire des différences hors champ.

Le pourcentage est `surface de mur référencée et revue / surface de mur référencée` ; les photos superposées sont fusionnées sans double compte. Les surfaces absentes des références, ainsi que les sols et plafonds, n’entrent pas dans ce calcul. Le pourcentage représente donc la **couverture des références murales**, et non celle du logement complet. Les photos non localisées ne sont pas comptées. Les approximations de surfaces en m² dépendent de la justesse du plan et de la calibration des références.

Les données sont persistées dans `data/housing/<logement>/references/<ref_id>/` et `data/housing/<logement>/scans/<scan_id>/` (photos, métadonnées, `coverage.json`, atlas, cartes et détections). Les anciens relevés restent consultables depuis le nouvel onglet ; le dernier scan colore également le plan principal. Ces dossiers sont ignorés par Git pour ne pas publier des photos de logement par inadvertance.

### Exemple reproductible

```bash
.venv/bin/python scripts/prepare_multiview_demo.py
.venv/bin/streamlit run app.py
```

Le script télécharge une [photographie d’un mur intérieur par Mia Gaitanidis](https://commons.wikimedia.org/wiki/File:Brick_wall,_inside.jpg), sous [licence CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), puis crée dans `examples/multiview/` des vues perspectivées synthétiques et une version avec une marque ajoutée numériquement. Cliquez sur **« Charger l’exemple multivue dans ce logement »** dans le nouvel onglet pour créer la référence et le scan automatiquement. Vous pouvez aussi ajouter `reference.jpg` sur `living_east` avec les valeurs de calibration par défaut, puis créer un scan avec **`scan_gauche.jpg` et `scan_marque.jpg`**. Le résultat attendu dans cette démonstration est environ 88 % de couverture et une variation visuelle près du centre du mur. `scan_droite.jpg` est une vue supplémentaire *sans* la marque ; elle sert à tester la couverture seule, et ne doit pas être mêlée à `scan_marque.jpg` si l’on veut simuler un seul état cohérent. Les images produites sont des transformations d’une même photographie : elles valident le recalage et le calcul, **pas** les performances sur des prises de vue indépendantes dans un logement réel.

Limites : une homographie est pertinente pour une face approximativement plane et des vues qui se recouvrent. Mobilier, occultations, surfaces uniformes, forts changements d’éclairage ou absence de recouvrement peuvent rendre la couverture ou les différences peu fiables. Les variations sont indicatives, pas une classification de dégradations. Pour une couverture réellement complète, il faudra des références pour tous les murs puis étendre la méthode aux sols/plafonds et, si nécessaire, à la profondeur ou à la pose caméra.

## Tests rapides

```bash
.venv/bin/python -m pytest -q
```
