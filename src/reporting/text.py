from __future__ import annotations

from collections import Counter
from typing import Any


TYPE_LABELS = {
    "crack": "fissure probable",
    "impact": "impact probable",
    "stain_or_dirt": "saleté ou tache probable",
    "object_change": "modification d’objet",
    "paint_peeling": "peinture écaillée probable",
    "unknown": "anomalie non classée",
}


def generate_text_report(report: dict[str, Any]) -> str:
    changes = report.get("detected_changes", [])
    if not changes:
        return ("État des lieux comparé automatiquement.\n\n"
                "Aucune différence visuelle significative n’a été détectée avec les paramètres actuels.\n\n"
                "Ce résultat reste une aide à la vérification visuelle.")
    types = Counter(change.get("type", "unknown") for change in changes)
    summary = ", ".join(f"{count} {TYPE_LABELS.get(kind, kind)}" for kind, count in types.items())
    lines = [
        "Rapport de comparaison des états des lieux",
        "=" * 44,
        "",
        f"{len(changes)} zone(s) présentant une différence visuelle ont été détectée(s).",
        f"Synthèse : {summary}.",
        f"Surface globale modifiée : {report.get('changed_surface_ratio', 0.0):.1%}.",
        "",
        "Détails :",
    ]
    for change in changes:
        kind = TYPE_LABELS.get(change.get("type", "unknown"), change.get("type", "unknown"))
        bbox = change.get("bbox", [0, 0, 0, 0])
        evidence = "; ".join(change.get("evidence", []))
        lines.extend([
            f"- Zone {change.get('id')}: {kind} sur {change.get('surface', 'surface inconnue')}",
            f"  Confiance : {change.get('confidence', 0.0):.0%} | Sévérité indicative : {change.get('severity', 'unknown')}",
            f"  Bounding box : ({bbox[0]}, {bbox[1]}) → ({bbox[2]}, {bbox[3]}) | Surface : {change.get('area_pixels', 0)} px",
            f"  Indices : {evidence or 'aucun indice détaillé'}",
        ])
    lines.extend([
        "",
        "Limites : ce rapport décrit des différences visuelles et ne constitue pas une expertise, une attribution de responsabilité ou une estimation financière.",
    ])
    return "\n".join(lines)

