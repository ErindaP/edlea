from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from .alignment.matcher import OpenCVMatcher, RegistrationBackend
from .comparison.feature import compute_feature_change_map
from .comparison.fusion import compute_fused_change_map
from .comparison.pixel import compute_pixel_maps
from .detection.regions import detect_changes, regional_statistics
from .features.dinov3 import DinoFeatureExtractor
from .image_io import load_image, match_size
from .segmentation.segmenter import HeuristicSegmenter, Segmenter
from .types import AlignmentResult
from .visualization.overlays import render_detections, render_heatmap, render_matches


def _default_config() -> dict[str, Any]:
    return {"image": {"max_size": 1280}, "alignment": {"method": "auto", "min_matches": 8, "ratio_test": 0.75},
            "features": {"backend": "auto", "model_name": "facebook/dinov2-base", "local_files_only": False, "image_size": 518},
            "comparison": {"dino_weight": 0.6, "ssim_weight": 0.3, "rgb_weight": 0.1},
            "detection": {"threshold": 0.4, "min_area": 150, "morph_kernel": 5}, "outputs": {"directory": "outputs"}}


class ChangeDetectionPipeline:
    def __init__(self, matcher: RegistrationBackend | None = None,
                 feature_extractor: DinoFeatureExtractor | None = None,
                 segmenter: Segmenter | None = None, comparator: Any | None = None,
                 config: dict[str, Any] | None = None):
        self.config = config or _default_config()
        alignment_cfg = self.config.get("alignment", {})
        features_cfg = self.config.get("features", {})
        matcher_kwargs = {key: alignment_cfg[key] for key in ("min_matches", "ratio_test", "method") if key in alignment_cfg}
        self.matcher = matcher or OpenCVMatcher(**matcher_kwargs)
        self.feature_extractor = feature_extractor or DinoFeatureExtractor(**features_cfg)
        self.segmenter = segmenter or HeuristicSegmenter()
        self.comparator = comparator

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ChangeDetectionPipeline":
        with open(path, encoding="utf-8") as handle:
            return cls(config=yaml.safe_load(handle))

    def compare(self, before: Any, after: Any, output_dir: str | Path | None = None) -> dict[str, Any]:
        max_size = self.config.get("image", {}).get("max_size", 1280)
        before_image, after_image = match_size(load_image(before, max_size), load_image(after, max_size))
        alignment: AlignmentResult = self.matcher.register(before_image, after_image)
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
        detections = detect_changes(fused_map, masks, threshold, detection_cfg.get("min_area", 150), detection_cfg.get("morph_kernel", 5))
        report = {"alignment": {"success": alignment.success, "method": "opencv_orb_homography", "num_matches": alignment.num_matches, "num_inliers": alignment.num_inliers},
                  "feature_backend": self.feature_extractor.backend_name,
                  "feature_backend_warning": getattr(self.feature_extractor, "load_error", None),
                  "global_change_score": round(float(fused_map.mean()), 6),
                  "changed_surface_ratio": round(float(np.mean(fused_map > threshold)), 6),
                  "regions": regional_statistics(fused_map, masks, threshold), "detected_changes": [item.to_dict() for item in detections]}
        visuals = {"before": before_image, "after": after_image, "aligned_before": alignment.aligned_before,
                   "matches": render_matches(before_image, after_image, alignment), "dino_heatmap": render_heatmap(dino_map, after_image),
                   "fused_heatmap": render_heatmap(fused_map, after_image), "detections": render_detections(after_image, detections)}
        result = {"report": report, "alignment": alignment, "change_maps": {"dino": dino_map, "fused": fused_map, **pixel_maps},
                  "masks": masks, "detections": detections, "visuals": visuals}
        if output_dir is not None:
            self.save_result(result, output_dir)
        return result

    @staticmethod
    def save_result(result: dict[str, Any], output_dir: str | Path) -> None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        names = {"aligned_before": "aligned_before.png", "dino_heatmap": "dino_heatmap.png", "fused_heatmap": "fused_heatmap.png", "detections": "detections.png", "matches": "matches.png"}
        for key, filename in names.items():
            Image.fromarray(result["visuals"][key]).save(directory / filename)
        (directory / "report.json").write_text(json.dumps(result["report"], indent=2), encoding="utf-8")
