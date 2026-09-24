from __future__ import annotations

import numpy as np

from src.alignment.keypoints import FallbackKeypointMatcher, KeypointMatches


class FakeMatcher:
    def __init__(self, name: str, count: int = 0, error: Exception | None = None):
        self.name = name
        self.count = count
        self.error = error

    def match(self, _image0, _image1, _mask=None):
        if self.error:
            raise self.error
        points = np.zeros((self.count, 2), np.float32)
        return KeypointMatches(points, points.copy(), np.ones(self.count, np.float32), self.name)


def test_auto_matcher_keeps_lightglue_when_it_has_enough_matches():
    matcher = FallbackKeypointMatcher(FakeMatcher("lightglue", 20), FakeMatcher("sift", 30))
    result = matcher.match(np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10, 3), np.uint8))
    assert result.backend == "lightglue"
    assert matcher.name == "lightglue"


def test_auto_matcher_uses_fallback_on_too_few_matches_or_error():
    weak = FallbackKeypointMatcher(FakeMatcher("lightglue", 5), FakeMatcher("sift", 16))
    failed = FallbackKeypointMatcher(FakeMatcher("lightglue", error=RuntimeError("inference")),
                                     FakeMatcher("sift", 14))
    image = np.zeros((10, 10, 3), np.uint8)

    assert weak.match(image, image).backend == "sift"
    assert failed.match(image, image).backend == "sift"
