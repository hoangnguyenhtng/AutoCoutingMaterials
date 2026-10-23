"""Build, save, load and update an embedding gallery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from product_retrieval.data import ManifestEntry, load_manifest


@dataclass
class Gallery:
    embeddings: np.ndarray
    metadata: list[dict[str, str]]
    model_info: dict[str, Any]
    created_at: str

    def __post_init__(self) -> None:
        self.embeddings = np.asarray(self.embeddings, dtype=np.float32)
        if self.embeddings.ndim != 2:
            raise ValueError("Gallery embeddings must be a 2-D matrix")
        if len(self.embeddings) != len(self.metadata):
            raise ValueError("Embedding and metadata counts do not match")
        if len(self.embeddings):
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            self.embeddings = self.embeddings / np.maximum(norms, 1e-12)

    @property
    def codes(self) -> list[str]:
        return sorted({item["product_code"] for item in self.metadata})


class GalleryBuilder:
    def __init__(self, encoder, gallery_path: str | Path = "data/gallery/gallery.npz", batch_size: int = 16):
        self.encoder = encoder
        self.gallery_path = Path(gallery_path)
        self.batch_size = batch_size

    def _from_entries(self, entries: Iterable[ManifestEntry]) -> Gallery:
        rows = list(entries)
        if not rows:
            raise ValueError("No gallery entries found in manifest")
        embeddings = self.encoder.encode_batch(
            [entry.image_path for entry in rows], batch_size=self.batch_size
        )
        return Gallery(
            embeddings=embeddings,
            metadata=[entry.as_metadata() for entry in rows],
            model_info=self.encoder.get_model_info(),
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def build(self, manifest_path: str | Path, gallery_path: str | Path | None = None) -> Gallery:
        gallery = self._from_entries(load_manifest(manifest_path, split="gallery"))
        self.save(gallery, gallery_path or self.gallery_path)
        return gallery

    @staticmethod
    def save(gallery: Gallery, gallery_path: str | Path) -> None:
        path = Path(gallery_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            embeddings=gallery.embeddings,
            metadata_json=np.array(json.dumps(gallery.metadata, ensure_ascii=False)),
            model_info_json=np.array(json.dumps(gallery.model_info, ensure_ascii=False)),
            created_at=np.array(gallery.created_at),
        )

    @staticmethod
    def load(gallery_path: str | Path) -> Gallery:
        path = Path(gallery_path)
        if not path.is_file():
            raise FileNotFoundError(f"Gallery does not exist: {path}")
        with np.load(path, allow_pickle=False) as archive:
            return Gallery(
                embeddings=archive["embeddings"],
                metadata=json.loads(str(archive["metadata_json"].item())),
                model_info=json.loads(str(archive["model_info_json"].item())),
                created_at=str(archive["created_at"].item()),
            )

    def update_codes(
        self,
        manifest_path: str | Path,
        product_codes: Iterable[str],
        gallery_path: str | Path | None = None,
    ) -> Gallery:
        path = Path(gallery_path or self.gallery_path)
        current = self.load(path)
        requested = set(product_codes)
        if not requested:
            raise ValueError("product_codes is empty")
        entries = [
            entry
            for entry in load_manifest(manifest_path, split="gallery")
            if entry.product_code in requested
        ]
        present = {entry.product_code for entry in entries}
        missing = requested - present
        if missing:
            raise ValueError(f"No gallery images found for codes: {', '.join(sorted(missing))}")
        replacement = self._from_entries(entries)
        compatibility_keys = ("model_name", "feature_type", "embedding_dimension", "preprocessing")
        incompatible = [
            key
            for key in compatibility_keys
            if current.model_info.get(key) != replacement.model_info.get(key)
        ]
        if incompatible:
            raise ValueError(
                "Cannot update a gallery built with a different encoder configuration: "
                + ", ".join(incompatible)
            )
        keep = np.array([item["product_code"] not in requested for item in current.metadata])
        embeddings = np.concatenate([current.embeddings[keep], replacement.embeddings], axis=0)
        metadata = [item for item, retain in zip(current.metadata, keep) if retain] + replacement.metadata
        updated = Gallery(
            embeddings=embeddings,
            metadata=metadata,
            model_info=replacement.model_info,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.save(updated, path)
        return updated
