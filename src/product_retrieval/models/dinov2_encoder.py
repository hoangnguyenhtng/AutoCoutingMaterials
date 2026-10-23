"""DINOv2 feature encoder with CLS and mean-patch representations."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from product_retrieval.preprocessing import ImagePreprocessor


MODEL_DIMENSIONS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


class DinoV2Encoder:
    """Extract L2-normalized embeddings from product crops.

    The model is loaded lazily on the first encode call so manifest and gallery
    utilities remain usable without a GPU or network connection.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        image_size: int = 224,
        device: str = "auto",
        feature_type: str = "cls",
        background_color: tuple[int, int, int] = (255, 255, 255),
        interpolation: str = "bicubic",
        model=None,
    ) -> None:
        if model_name not in MODEL_DIMENSIONS:
            raise ValueError(f"Unsupported model {model_name!r}; choose from {sorted(MODEL_DIMENSIONS)}")
        if feature_type not in {"cls", "patch_mean"}:
            raise ValueError("feature_type must be 'cls' or 'patch_mean'")
        self.model_name = model_name
        self.feature_type = feature_type
        self.requested_device = device
        self.preprocessor = ImagePreprocessor(
            image_size=image_size,
            background_color=background_color,
            interpolation=interpolation,
        )
        self._model = model
        self._device = None

    @property
    def device(self) -> str:
        if self._device is None:
            import torch

            if self.requested_device == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self.requested_device
        return self._device

    def _ensure_model(self):
        if self._model is None:
            import torch

            self._model = torch.hub.load("facebookresearch/dinov2", self.model_name)
        self._model = self._model.to(self.device).eval()
        return self._model

    def _features(self, batch):
        import torch
        import torch.nn.functional as functional

        model = self._ensure_model()
        with torch.inference_mode():
            output = model.forward_features(batch.to(self.device))
            if not isinstance(output, dict):
                raise RuntimeError("DINOv2 forward_features returned an unsupported result")
            key = "x_norm_clstoken" if self.feature_type == "cls" else "x_norm_patchtokens"
            if key not in output:
                raise RuntimeError(f"DINOv2 output is missing {key}")
            features = output[key]
            if self.feature_type == "patch_mean":
                features = features.mean(dim=1)
            features = functional.normalize(features.float(), p=2, dim=1)
        return features.cpu().numpy().astype(np.float32, copy=False)

    def encode_image(self, image_path: str | Path) -> np.ndarray:
        """Extract one normalized embedding with shape ``(dimension,)``."""

        tensor = self.preprocessor(image_path).unsqueeze(0)
        return self._features(tensor)[0]

    def encode_batch(
        self,
        image_paths: Iterable[str | Path],
        batch_size: int = 16,
    ) -> np.ndarray:
        """Extract embeddings while bounding inference memory use."""

        import torch

        paths = list(image_paths)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not paths:
            return np.empty((0, MODEL_DIMENSIONS[self.model_name]), dtype=np.float32)
        chunks: list[np.ndarray] = []
        for start in range(0, len(paths), batch_size):
            tensors = [self.preprocessor(path) for path in paths[start : start + batch_size]]
            chunks.append(self._features(torch.stack(tensors)))
        return np.concatenate(chunks, axis=0)

    def get_model_info(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "embedding_dimension": MODEL_DIMENSIONS[self.model_name],
            "feature_type": self.feature_type,
            "device": self.device,
            "preprocessing": self.preprocessor.get_config(),
        }

