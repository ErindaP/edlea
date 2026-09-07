from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np

from ..types import AlignmentResult
from .warp import warp_image


class RegistrationBackend(ABC):
    """Registration abstraction, replaceable by RGB-D/SLAM later."""

    @abstractmethod
    def register(self, before: np.ndarray, after: np.ndarray) -> AlignmentResult:
        raise NotImplementedError


class OpenCVMatcher(RegistrationBackend):
    """Fast local fallback using ORB, ratio-test matching and RANSAC."""

    def __init__(self, min_matches: int = 8, ratio_test: float = 0.75, method: str = "auto"):
        self.min_matches = min_matches
        self.ratio_test = ratio_test
        self.method = method

    def register(self, before: np.ndarray, after: np.ndarray) -> AlignmentResult:
        height, width = after.shape[:2]
        identity = np.eye(3, dtype=np.float64)
        gray_before = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
        gray_after = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY)
        orb = cv2.ORB_create(nfeatures=3000, fastThreshold=7)
        key_before, desc_before = orb.detectAndCompute(gray_before, None)
        key_after, desc_after = orb.detectAndCompute(gray_after, None)
        if desc_before is None or desc_after is None:
            aligned, valid = warp_image(before, identity, (height, width))
            return AlignmentResult(aligned, identity, valid_mask=valid)
        raw = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_before, desc_after, k=2)
        good = [pair[0] for pair in raw if len(pair) == 2 and pair[0].distance < self.ratio_test * pair[1].distance]
        points_before = np.float32([key_before[m.queryIdx].pt for m in good]) if good else np.empty((0, 2), np.float32)
        points_after = np.float32([key_after[m.trainIdx].pt for m in good]) if good else np.empty((0, 2), np.float32)
        confidence = np.array([1.0 / (1.0 + m.distance) for m in good], dtype=np.float32)
        if len(good) < self.min_matches:
            aligned, valid = warp_image(before, identity, (height, width))
            return AlignmentResult(aligned, identity, points_before, points_after, confidence, valid_mask=valid)
        homography, inliers = cv2.findHomography(points_before, points_after, cv2.RANSAC, 5.0)
        if homography is None or inliers is None or int(inliers.sum()) < 4:
            homography = identity
            inlier_mask = np.zeros(len(good), dtype=bool)
            success = False
        else:
            inlier_mask = inliers.ravel().astype(bool)
            success = True
        aligned, valid = warp_image(before, homography, (height, width))
        return AlignmentResult(aligned, homography, points_before, points_after, confidence, inlier_mask, success, valid)

