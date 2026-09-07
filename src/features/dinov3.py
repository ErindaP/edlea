from __future__ import annotations

import logging
from typing import Any
import os

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

logger = logging.getLogger(__name__)


class DinoFeatureExtractor:
    """DINOv2/DINOv3 dense features with a deterministic local fallback."""

    def __init__(self, model_name: str, backend: str = "auto", image_size: int = 518,
                 local_files_only: bool = False):
        self.model_name = model_name
        self.backend = backend
        self.image_size = image_size
        self.local_files_only = local_files_only
        self.processor: Any = None
        self.model: Any = None
        self.load_error: str | None = None
        self.device = "cpu"
        if backend in {"auto", "dino", "dinov2", "dinov3"}:
            self._try_load_model()

    @property
    def backend_name(self) -> str:
        if self.model is None:
            return "local-descriptor"
        model_name = self.model_name.lower()
        return "dinov2" if "dinov2" in model_name else "dinov3" if "dinov3" in model_name else "dino"

    def _try_load_model(self) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel

            token = os.getenv("HF_TOKEN")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            kwargs = {"local_files_only": self.local_files_only}
            if token:
                kwargs["token"] = token
            self.processor = AutoImageProcessor.from_pretrained(self.model_name, **kwargs)
            self.model = AutoModel.from_pretrained(self.model_name, **kwargs).to(self.device).eval()
            logger.info("Loaded DINO model %s on %s", self.model_name, self.device)
        except Exception as exc:
            self.load_error = str(exc)
            logger.warning("DINO unavailable (%s); using local fallback", exc)
            self.processor = None
            self.model = None

    def extract(self, image: np.ndarray) -> np.ndarray:
        return self._extract_dino(image) if self.model is not None else self._extract_local(image)

    def _extract_dino(self, image: np.ndarray) -> np.ndarray:
        import torch
        inputs = self.processor(images=image, return_tensors="pt",
                                size={"shortest_edge": self.image_size}, do_center_crop=False)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            try:
                output = self.model(**inputs, interpolate_pos_encoding=True)
            except TypeError:
                output = self.model(**inputs)
        tokens = getattr(output, "last_hidden_state", None)
        if tokens is None:
            tokens = output[0]
        token_count = int(tokens.shape[1])
        patch_size = int(getattr(self.model.config, "patch_size", 14))
        input_height, input_width = inputs["pixel_values"].shape[-2:]
        grid_height, grid_width = input_height // patch_size, input_width // patch_size
        patch_count = grid_height * grid_width
        if patch_count == 0 or patch_count > token_count:
            raise RuntimeError("DINO output does not contain patch tokens")
        # Drop a possible CLS token and register tokens from the prefix.
        tokens = tokens[:, token_count - patch_count:, :]
        features = tokens.reshape(1, grid_height, grid_width, -1)
        features = torch.nn.functional.normalize(features, dim=-1)
        return features[0].detach().cpu().numpy().astype(np.float32)

    @staticmethod
    def _extract_local(image: np.ndarray) -> np.ndarray:
        """Dense normalized colour/gradient descriptor used without model weights."""
        small = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)
        angle = cv2.phase(gx, gy, angleInDegrees=False)
        descriptor = np.concatenate([small, gray[..., None], gx[..., None], gy[..., None], magnitude[..., None],
                                      np.sin(angle)[..., None], np.cos(angle)[..., None]], axis=-1)
        norm = np.linalg.norm(descriptor, axis=-1, keepdims=True)
        return (descriptor / np.maximum(norm, 1e-6)).astype(np.float32)
