from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from .alignment.coverage import analyze_pair_coverage
from .alignment.matcher import OpenCVMatcher, RegistrationBackend
from .comparison.feature import compute_feature_change_map
from .comparison.fusion import compute_fused_change_map
from .comparison.pixel import compute_pixel_maps
from .detection.regions import detect_changes, regional_statistics
from .detection.classification import AnomalyClassifier
from .features.dinov3 import DinoFeatureExtractor
from .image_io import load_image, match_size
from .reporting.text import generate_text_report
from .segmentation.segmenter import HeuristicSegmenter, Segmenter
from .visualization.overlays import render_detections, render_heatmap, render_matches


def _default_config() -> dict[str, Any]:
    return {"image": {"max_size": 1280}, "alignment": {"method": "auto", "min_matches": 8, "ratio_test": 0.75},
            "features": {"backend": "auto", "model_name": "facebook/dinov2-base", "local_files_only": False, "image_size": 518},
            "comparison": {"dino_weight": 0.6, "ssim_weight": 0.3, "rgb_weight": 0.1},
            "detection": {"threshold": 0.3, "min_area": 100, "morph_kernel": 5, "opening_kernel": 0,
                          "hysteresis_ratio": 0.35},
            "outputs": {"directory": "outputs"}}


class ChangeDetectionPipeline:
    def __init__(self, matcher: RegistrationBackend | None = None,
                 feature_extractor: DinoFeatureExtractor | None = None,
                 segmenter: Segmenter | None = None, classifier: AnomalyClassifier | None = None,
                 comparator: Any | None = None,
                 config: dict[str, Any] | None = None):
        self.config = config or _default_config()
        alignment_cfg = self.config.get("alignment", {})
        features_cfg = self.config.get("features", {})
        matcher_kwargs = {key: alignment_cfg[key] for key in ("min_matches", "ratio_test", "method") if key in alignment_cfg}
        self.matcher = matcher or OpenCVMatcher(**matcher_kwargs)
        self.feature_extractor = feature_extractor or DinoFeatureExtractor(**features_cfg)
        self.segmenter = segmenter or HeuristicSegmenter()
        self.classifier = classifier or AnomalyClassifier()
        self.comparator = comparator

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ChangeDetectionPipeline":
        with open(path, encoding="utf-8") as handle:
            return cls(config=yaml.safe_load(handle))

    def compare(self, before: Any, after: Any, output_dir: str | Path | None = None,
                *, coverage_analysis: bool = False) -> dict[str, Any]:
        max_size = self.config.get("image", {}).get("max_size", 1280)
        before_image = load_image(before, max_size)
        after_image = load_image(after, max_size)
        coverage_result = None
        if coverage_analysis:
            coverage_result = analyze_pair_coverage(before_image, after_image)
            alignment = coverage_result.alignment
        else:
            before_image, after_image = match_size(before_image, after_image)
            alignment = self.matcher.register(before_image, after_image)
        features_before = self.feature_extractor.extract(alignment.aligned_before)
        features_after = self.feature_extractor.extract(after_image)
        dino_map = compute_feature_change_map(features_before, features_after, after_image.shape[:2])
        pixel_maps = compute_pixel_maps(alignment.aligned_before, after_image)
        comparison_cfg = self.config.get("comparison", {})
        fused_map = compute_fused_change_map(dino_map, pixel_maps["ssim"], pixel_maps["rgb"],
                                             comparison_cfg.get("dino_weight", 0.6), comparison_cfg.get("ssim_weight", 0.3),
                                             comparison_cfg.get("rgb_weight", 0.1))
        if alignment.valid_mask is not None:
            fused_map = np.where(alignment.valid_mask, fused_map, 0.0)
        masks = self.segmenter.segment(after_image)
        detection_cfg = self.config.get("detection", {})
        threshold = detection_cfg.get("threshold", 0.4)
        hysteresis_ratio = detection_cfg.get("hysteresis_ratio", 0.35)
        if coverage_result and coverage_result.support_mode == "inlier_supported":
            # Weakly supported homographies retain more residual viewpoint
            # noise. Do not let low-score pixels connect it into giant boxes.
            hysteresis_ratio = max(hysteresis_ratio, 0.65)
        detections = detect_changes(fused_map, masks, threshold, detection_cfg.get("min_area", 100),
                                    detection_cfg.get("morph_kernel", 5), detection_cfg.get("opening_kernel", 0),
                                    hysteresis_ratio,
                                    alignment.valid_mask if coverage_analysis else None)
        classified_changes = []
        for detection in detections:
            classification = self.classifier.classify(alignment.aligned_before, after_image, fused_map, detection)
            item = detection.to_dict()
            item.update(classification.to_dict())
            classified_changes.append(item)
        comparable_values = fused_map[alignment.valid_mask] if coverage_analysis else fused_map.ravel()
        report = {"alignment": {"success": alignment.success,
                                "method": coverage_result.matching_backend if coverage_result else "opencv_orb_homography",
                                "num_matches": alignment.num_matches, "num_inliers": alignment.num_inliers},
                  "feature_backend": self.feature_extractor.backend_name,
                  "feature_backend_warning": getattr(self.feature_extractor, "load_error", None),
                  "global_change_score": round(float(comparable_values.mean()), 6),
                  "max_change_score": round(float(comparable_values.max()), 6),
                  "detection_threshold": threshold,
                  "detection_hysteresis_ratio": hysteresis_ratio,
                  "changed_surface_ratio": round(float(np.mean(comparable_values > threshold)), 6),
                  "regions": regional_statistics(
                      fused_map, masks, threshold,
                      alignment.valid_mask if coverage_analysis else None,
                  ), "detected_changes": classified_changes}
        if coverage_result:
            report["pair_coverage"] = coverage_result.to_dict()
        text_report = generate_text_report(report)
        visuals = {"before": before_image, "after": after_image, "aligned_before": alignment.aligned_before,
                   "matches": render_matches(before_image, after_image, alignment), "dino_heatmap": render_heatmap(dino_map, after_image),
                   "fused_heatmap": render_heatmap(fused_map, after_image), "detections": render_detections(after_image, detections)}
        if coverage_result:
            visuals["coverage_overlay"] = coverage_result.overlay
        result = {"report": report, "alignment": alignment, "change_maps": {"dino": dino_map, "fused": fused_map, **pixel_maps},
                  "masks": masks, "detections": detections, "visuals": visuals, "text_report": text_report,
                  "source_images": {"aligned_before": alignment.aligned_before, "after": after_image}}
        if coverage_result:
            result["coverage_status"] = coverage_result.wall_status
        if output_dir is not None:
            self.save_result(result, output_dir)
        return result

    @staticmethod
    def save_result(result: dict[str, Any], output_dir: str | Path) -> None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        names = {"aligned_before": "aligned_before.png", "dino_heatmap": "dino_heatmap.png",
                 "fused_heatmap": "fused_heatmap.png", "detections": "detections.png", "matches": "matches.png",
                 "coverage_overlay": "coverage_overlay.png"}
        for key, filename in names.items():
            if key in result["visuals"]:
                Image.fromarray(result["visuals"][key]).save(directory / filename)
        if "coverage_status" in result:
            Image.fromarray(result["coverage_status"]).save(directory / "coverage_status.png")
        (directory / "report.json").write_text(json.dumps(result["report"], indent=2), encoding="utf-8")
        (directory / "report.txt").write_text(result["text_report"], encoding="utf-8")
        crops_dir = directory / "anomalies"
        crops_dir.mkdir(exist_ok=True)
        for detection in result["detections"]:
            x1, y1, x2, y2 = detection.bbox
            identifier = f"change_{detection.id}"
            Image.fromarray(result["source_images"]["aligned_before"][y1:y2, x1:x2]).save(crops_dir / f"{identifier}_before.png")
            Image.fromarray(result["source_images"]["after"][y1:y2, x1:x2]).save(crops_dir / f"{identifier}_after.png")
            if detection.binary_mask is not None:
                Image.fromarray((detection.binary_mask[y1:y2, x1:x2].astype(np.uint8) * 255)).save(crops_dir / f"{identifier}_mask.png")
