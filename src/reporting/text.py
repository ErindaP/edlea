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
    coverage = report.get("pair_coverage")
    coverage_text = ""
    if coverage:
        coverage_text = (
            "\nAnalyse de couverture : "
            f"{coverage.get('coverage_of_before_percent', 0):.1f} % de l’image Avant retrouvée ; "
            f"{coverage.get('comparable_after_percent', 0):.1f} % de l’image Après analysée. "
            "Les zones hors recouvrement ont été exclues.\n"
        )
    if not changes:
        max_score = report.get("max_change_score", 0.0)
        threshold = report.get("detection_threshold", 0.0)
        if max_score >= threshold:
            detail = (f"Des pixels présentent pourtant un score de changement élevé (maximum {max_score:.2f}, "
                      f"seuil {threshold:.2f}), mais aucune composante n’a dépassé le filtre de surface ou de morphologie.")
        else:
            detail = f"Le score maximal observé est {max_score:.2f}, inférieur au seuil de {threshold:.2f}."
        deterministic = ("État des lieux comparé automatiquement.\n\n"
                "Aucune zone de différence exploitable n’a été détectée avec les paramètres actuels.\n"
                f"{detail}\n{coverage_text}\n"
                "Ce résultat reste une aide à la vérification visuelle.")
        return _append_global_analysis(deterministic, report)
    types = Counter(change.get("type", "unknown") for change in changes)
    summary = ", ".join(f"{count} {TYPE_LABELS.get(kind, kind)}" for kind, count in types.items())
    lines = [
        "Rapport de comparaison des états des lieux",
        "=" * 44,
        "",
        f"{len(changes)} zone(s) présentant une différence visuelle ont été détectée(s).",
        f"Synthèse : {summary}.",
        f"Surface globale modifiée : {report.get('changed_surface_ratio', 0.0):.1%}.",
        coverage_text.strip(),
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
    return _append_global_analysis("\n".join(lines), report)


def _append_global_analysis(text: str, report: dict[str, Any]) -> str:
    analysis = report.get("global_visual_analysis", {})
    if analysis.get("status") != "success" or not analysis.get("text"):
        return text
    return (
        f"{text}\n\n"
        "Analyse visuelle globale — modèle local\n"
        "=" * 40 + "\n\n"
        f"{analysis['text']}\n\n"
        "Cette synthèse générée doit être confirmée par une vérification humaine."
    )
