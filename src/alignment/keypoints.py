"""Interchangeable local-feature matchers for multiview registration."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from typing import Protocol

import cv2
import numpy as np
from PIL import Image


LIGHTGLUE_MODEL = "ETH-CVG/lightglue_superpoint"


@dataclass(frozen=True)
class KeypointMatches:
    points0: np.ndarray
    points1: np.ndarray
    confidence: np.ndarray
    backend: str

    @classmethod
    def empty(cls, backend: str) -> "KeypointMatches":
        return cls(np.empty((0, 2), np.float32), np.empty((0, 2), np.float32),
                   np.empty(0, np.float32), backend)


class KeypointMatcher(Protocol):
    name: str

    def match(self, image0: np.ndarray, image1: np.ndarray,
              image1_mask: np.ndarray | None = None) -> KeypointMatches:
        """Return points in image0 and image1 coordinates."""


class OpenCVKeypointMatcher:
    """Local fallback used when the learned matcher is unavailable."""

    name = "sift_bf"

    def match(self, image0: np.ndarray, image1: np.ndarray,
              image1_mask: np.ndarray | None = None) -> KeypointMatches:
        gray0 = cv2.cvtColor(image0, cv2.COLOR_RGB2GRAY)
        gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        if hasattr(cv2, "SIFT_create"):
            detector = cv2.SIFT_create(nfeatures=3500)
            norm = cv2.NORM_L2
        else:
            detector = cv2.ORB_create(nfeatures=4000, fastThreshold=7)
            norm = cv2.NORM_HAMMING
            self.name = "orb_bf"
        keys0, descriptors0 = detector.detectAndCompute(gray0, None)
        keys1, descriptors1 = detector.detectAndCompute(gray1, image1_mask)
        if descriptors0 is None or descriptors1 is None:
            return KeypointMatches.empty(self.name)
        pairs = cv2.BFMatcher(norm).knnMatch(descriptors0, descriptors1, k=2)
        good = [first for pair in pairs if len(pair) == 2 for first, second in [pair]
                if first.distance < 0.72 * second.distance]
        if not good:
            return KeypointMatches.empty(self.name)
        points0 = np.float32([keys0[match.queryIdx].pt for match in good])
        points1 = np.float32([keys1[match.trainIdx].pt for match in good])
        confidence = np.float32([1.0 / (1.0 + match.distance) for match in good])
        return KeypointMatches(points0, points1, confidence, self.name)


class SuperPointLightGlueMatcher:
    """Official Transformers implementation of SuperPoint + LightGlue."""

    name = "superpoint_lightglue"

    def __init__(self, model_name: str = LIGHTGLUE_MODEL, threshold: float = 0.1):
        import torch
        from transformers import AutoImageProcessor, AutoModelForKeypointMatching

        self.torch = torch
        self.threshold = float(threshold)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        token = os.getenv("HF_TOKEN") or None
        self.processor = AutoImageProcessor.from_pretrained(model_name, token=token)
        self.model = AutoModelForKeypointMatching.from_pretrained(model_name, token=token)
        self.model.eval().to(self.device)

    def match(self, image0: np.ndarray, image1: np.ndarray,
              image1_mask: np.ndarray | None = None) -> KeypointMatches:
        images = [Image.fromarray(image0), Image.fromarray(image1)]
        inputs = self.processor(images, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        sizes = [[(image.height, image.width) for image in images]]
        result = self.processor.post_process_keypoint_matching(
            outputs, sizes, threshold=self.threshold,
        )[0]
        points0 = result["keypoints0"].detach().float().cpu().numpy().astype(np.float32)
        points1 = result["keypoints1"].detach().float().cpu().numpy().astype(np.float32)
        confidence = result["matching_scores"].detach().float().cpu().numpy().astype(np.float32)
        if image1_mask is not None and len(points1):
            x = np.clip(np.rint(points1[:, 0]).astype(int), 0, image1_mask.shape[1] - 1)
            y = np.clip(np.rint(points1[:, 1]).astype(int), 0, image1_mask.shape[0] - 1)
            keep = image1_mask[y, x] > 0
            points0, points1, confidence = points0[keep], points1[keep], confidence[keep]
        return KeypointMatches(points0, points1, confidence, self.name)


class FallbackKeypointMatcher:
    """Use the learned matcher first and transparently recover on CPU/OpenCV."""

    def __init__(self, primary: KeypointMatcher, fallback: KeypointMatcher):
        self.primary = primary
        self.fallback = fallback
        self.name = primary.name

    def match(self, image0: np.ndarray, image1: np.ndarray,
              image1_mask: np.ndarray | None = None) -> KeypointMatches:
        try:
            result = self.primary.match(image0, image1, image1_mask)
            if len(result.points0) >= 12:
                self.name = self.primary.name
                return result
            fallback_result = self.fallback.match(image0, image1, image1_mask)
            if len(fallback_result.points0) > len(result.points0):
                self.name = self.fallback.name
                return fallback_result
            self.name = self.primary.name
            return result
        except Exception:
            self.name = self.fallback.name
            return self.fallback.match(image0, image1, image1_mask)


@lru_cache(maxsize=3)
def get_keypoint_matcher(backend: str = "auto") -> KeypointMatcher:
    backend = backend.lower().strip()
    if backend in {"opencv", "sift", "orb"}:
        return OpenCVKeypointMatcher()
    if backend not in {"auto", "lightglue", "superpoint_lightglue"}:
        raise ValueError(f"Backend d’appariement inconnu : {backend}")
    try:
        lightglue = SuperPointLightGlueMatcher()
        return FallbackKeypointMatcher(lightglue, OpenCVKeypointMatcher()) if backend == "auto" else lightglue
    except Exception:
        if backend != "auto":
            raise
        return OpenCVKeypointMatcher()
