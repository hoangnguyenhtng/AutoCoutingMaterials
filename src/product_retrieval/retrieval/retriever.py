"""Exact cosine-similarity retrieval for small and medium galleries."""

from __future__ import annotations

import numpy as np

from .gallery_builder import Gallery


class ProductRetriever:
    def __init__(
        self,
        gallery: Gallery,
        aggregation: str = "top_mean",
        references_per_code: int = 3,
    ) -> None:
        if aggregation not in {"top_mean", "max", "centroid"}:
            raise ValueError("aggregation must be top_mean, max or centroid")
        if references_per_code <= 0:
            raise ValueError("references_per_code must be positive")
        self.gallery = gallery
        self.aggregation = aggregation
        self.references_per_code = references_per_code
        self._indices_by_code: dict[str, np.ndarray] = {}
        for code in gallery.codes:
            self._indices_by_code[code] = np.array(
                [i for i, row in enumerate(gallery.metadata) if row["product_code"] == code]
            )
        self._centroids: dict[str, np.ndarray] = {}
        for code, indices in self._indices_by_code.items():
            centroid = gallery.embeddings[indices].mean(axis=0)
            self._centroids[code] = centroid / max(float(np.linalg.norm(centroid)), 1e-12)

    def _normalize_query(self, query_embedding) -> np.ndarray:
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.gallery.embeddings.shape[1]:
            raise ValueError(
                f"Query dimension {query.shape[0]} does not match gallery "
                f"dimension {self.gallery.embeddings.shape[1]}"
            )
        return query / max(float(np.linalg.norm(query)), 1e-12)

    def search(self, query_embedding, top_k: int = 5) -> list[dict[str, object]]:
        query = self._normalize_query(query_embedding)
        top_k = max(0, min(int(top_k), len(self.gallery.embeddings)))
        scores = self.gallery.embeddings @ query
        indices = np.argsort(-scores, kind="stable")[:top_k]
        return [
            {"rank": rank, "score": float(scores[index]), **self.gallery.metadata[index]}
            for rank, index in enumerate(indices, start=1)
        ]

    def all_code_scores(self, query_embedding) -> dict[str, float]:
        query = self._normalize_query(query_embedding)
        image_scores = self.gallery.embeddings @ query
        result: dict[str, float] = {}
        for code, indices in self._indices_by_code.items():
            if self.aggregation == "centroid":
                score = float(self._centroids[code] @ query)
            elif self.aggregation == "max":
                score = float(image_scores[indices].max())
            else:
                values = np.sort(image_scores[indices])[-self.references_per_code :]
                score = float(values.mean())
            result[code] = score
        return result

    def rank_codes(self, query_embedding, top_k: int = 5) -> list[dict[str, object]]:
        scores = self.all_code_scores(query_embedding)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[: max(0, int(top_k))]
        return [
            {"rank": rank, "product_code": code, "score": score}
            for rank, (code, score) in enumerate(ranked, start=1)
        ]
