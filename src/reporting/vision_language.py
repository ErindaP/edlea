from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any

import cv2
import numpy as np
from PIL import Image


DEFAULT_VLM_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
REPORT_TYPE_LABELS = {
    "crack": "une fissure",
    "impact": "un impact",
    "stain_or_dirt": "une tache ou saleté",
    "object_change": "une modification d’objet",
    "paint_peeling": "une dégradation de peinture",
    "unknown": "une anomalie visuelle",
}


def build_visual_report_prompt(report: dict[str, Any]) -> str:
    changes = report.get("detected_changes", [])
    categories = Counter(str(item.get("type", "unknown")) for item in changes)
    priority = {"high": 2, "medium": 1, "low": 0}
    prominent = sorted(
        changes,
        key=lambda item: (
            priority.get(str(item.get("severity", "low")), 0),
            float(item.get("confidence", 0) or 0),
            int(item.get("area_pixels", 0) or 0),
        ),
        reverse=True,
    )[:3]
    detector_context = {
        "nombre_total_de_zones": len(changes),
        "types_et_effectifs": dict(categories),
        "surface_modifiee_fraction": report.get("changed_surface_ratio"),
        "trois_exemples_prioritaires": [
            {
                "type": item.get("type", "unknown"),
                "surface": item.get("surface", "unknown"),
                "confiance": item.get("confidence"),
            }
            for item in prominent
        ],
    }
    return (
        "État des lieux : première image AVANT, seconde image APRÈS. Compare globalement les dommages visibles. "
        "Vérifie si un défaut était déjà visible dans les deux images ; distingue nouveau, disparu ou modifié. "
        "Ignore lumière et cadrage. Ne déduis ni cause, ni profondeur, ni responsabilité. "
        "Une très faible surface modifiée indique seulement une variation locale incertaine. "
        "Indices du détecteur (non exhaustifs, à vérifier visuellement) :\n"
        f"{json.dumps(detector_context, ensure_ascii=False)}\n"
        "Réponds en français en 70 mots maximum : une phrase pour le constat, une pour l'évolution, "
        "une pour la vérification. Ne détaille pas chaque boîte."
    )


def _resize_for_vlm(image: np.ndarray, max_side: int = 768) -> Image.Image:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    return Image.fromarray(image.astype(np.uint8))


def guard_visual_report(text: str, report: dict[str, Any]) -> str:
    """Remove non-observable claims and temper tiny localized differences."""
    # Depth and structural soundness cannot be inferred from these 2D images.
    safe_text = re.sub(
        r"[^\n.!?]*(?:profondeur|profonde|solidité|structurelle)[^\n.!?]*[.!?]?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    safe_text = re.sub(r"[ \t]{2,}", " ", safe_text).strip()
    changed_ratio = float(report.get("changed_surface_ratio", 0.0) or 0.0)
    changes = report.get("detected_changes", [])
    if changed_ratio >= 0.002 or not changes:
        words = safe_text.split()
        return " ".join(words[:80]) + ("…" if len(words) > 80 else "") if words else "L’analyse globale n’a pas produit de constat exploitable."

    labels = []
    for change in changes:
        label = REPORT_TYPE_LABELS.get(change.get("type", "unknown"), "une anomalie visuelle")
        if label not in labels:
            labels.append(label)
    observed = ", ".join(labels)
    lower = text.lower()
    visible_in_both = (
        "avant" in lower
        and "après" in lower
        and any(marker in lower for marker in ("identique", "deux images", "deux états", "déjà visible"))
    )
    if visible_in_both:
        global_statement = f"{observed.capitalize()} est visible dans les deux états comparés."
    else:
        global_statement = f"L’analyse croisée signale {observed} dans la scène comparée."
    return (
        f"Constat global : {global_statement}\n\n"
        "Évolutions observées : Le détecteur mesure une variation visuelle localisée sur une portion du défaut. "
        "La faible surface modifiée ne permet pas de conclure automatiquement à une apparition ou à une aggravation globale.\n\n"
        "Vérification recommandée : Contrôler visuellement la zone signalée et comparer son étendue avec les clichés d’origine."
    )


class LocalVisionReporter:
    """Lazy local VLM used only to phrase a global visual comparison."""

    def __init__(self, model_name: str = DEFAULT_VLM_MODEL, local_files_only: bool = False,
                 max_new_tokens: int = 120):
        self.model_name = model_name
        self.local_files_only = bool(local_files_only)
        self.max_new_tokens = int(max_new_tokens)
        self.model: Any | None = None
        self.processor: Any | None = None
        self.device = "cpu"

    def _load(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        token = os.getenv("HF_TOKEN") or None
        common = {"local_files_only": self.local_files_only}
        if token:
            common["token"] = token
        self.processor = AutoProcessor.from_pretrained(self.model_name, **common)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_name,
            dtype=dtype,
            **common,
        ).to(self.device)
        self.model.eval()

    def analyze(self, before: np.ndarray, after: np.ndarray, report: dict[str, Any]) -> dict[str, Any]:
        try:
            self._load()
            import torch

            images = [_resize_for_vlm(before), _resize_for_vlm(after)]
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": images[0]},
                    {"type": "image", "image": images[1]},
                    {"type": "text", "text": build_visual_report_prompt(report)},
                ],
            }]
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device)
            input_length = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            raw_text = self.processor.batch_decode(
                output_ids[:, input_length:],
                skip_special_tokens=True,
            )[0].strip()
            if not raw_text:
                raise RuntimeError("Le modèle local a produit une réponse vide.")
            return {
                "status": "success",
                "model": self.model_name,
                "device": self.device,
                "text": guard_visual_report(raw_text, report),
                "raw_text": raw_text,
            }
        except Exception as exc:  # The deterministic report must remain available.
            return {
                "status": "error",
                "model": self.model_name,
                "device": self.device,
                "text": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
