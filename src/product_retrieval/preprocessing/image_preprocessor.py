"""Aspect-ratio preserving preprocessing for DINOv2."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageOps


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImagePreprocessor:
    def __init__(
        self,
        image_size: int = 224,
        background_color: Sequence[int] = (255, 255, 255),
        interpolation: str = "bicubic",
    ) -> None:
        if image_size <= 0 or image_size % 14 != 0:
            raise ValueError("image_size must be a positive multiple of DINOv2 patch size 14")
        if len(background_color) != 3:
            raise ValueError("background_color must contain three RGB values")
        self.image_size = int(image_size)
        self.background_color = tuple(int(v) for v in background_color)
        modes = {"nearest": Image.Resampling.NEAREST, "bilinear": Image.Resampling.BILINEAR,
                 "bicubic": Image.Resampling.BICUBIC, "lanczos": Image.Resampling.LANCZOS}
        try:
            self.resample = modes[interpolation.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported interpolation: {interpolation}") from exc
        self.interpolation = interpolation.lower()

    def prepare_pil(self, image: str | Path | Image.Image) -> Image.Image:
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                rgb = ImageOps.exif_transpose(opened).convert("RGB")
        elif isinstance(image, Image.Image):
            rgb = ImageOps.exif_transpose(image).convert("RGB")
        else:
            raise TypeError("image must be a path or PIL.Image.Image")

        width, height = rgb.size
        scale = min(self.image_size / width, self.image_size / height)
        resized_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        resized = rgb.resize(resized_size, self.resample)
        canvas = Image.new("RGB", (self.image_size, self.image_size), self.background_color)
        offset = ((self.image_size - resized.width) // 2, (self.image_size - resized.height) // 2)
        canvas.paste(resized, offset)
        return canvas

    def __call__(self, image: str | Path | Image.Image):
        """Return a normalized CHW float32 torch.Tensor."""

        import torch

        prepared = self.prepare_pil(image)
        array = np.asarray(prepared, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std

    def get_config(self) -> dict[str, object]:
        return {
            "image_size": self.image_size,
            "background_color": list(self.background_color),
            "interpolation": self.interpolation,
        }
