import json
from pathlib import Path

import numpy as np

from product_retrieval.data import ManifestEntry
from product_retrieval.evaluation import Evaluator
from product_retrieval.retrieval import Gallery, ProductRetriever


class FakeEncoder:
    def encode_batch(self, paths, batch_size=16):
        vectors = {
            "known_a.jpg": [1.0, 0.0],
            "known_b.jpg": [0.0, 1.0],
            "unknown.jpg": [-1.0, -1.0],
        }
        return np.asarray([vectors[Path(path).name] for path in paths], dtype=np.float32)


def test_evaluator_metrics_and_reports(tmp_path):
    gallery = Gallery(
        embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        metadata=[
            {"image_path": "gallery_a.jpg", "product_code": "A"},
            {"image_path": "gallery_b.jpg", "product_code": "B"},
        ],
        model_info={},
        created_at="now",
    )
    retriever = ProductRetriever(gallery, aggregation="max")
    evaluator = Evaluator(
        FakeEncoder(),
        retriever,
        {"match_score": 0.8, "mismatch_score": 0.8, "unknown_score": 0.2, "margin": 0.1},
    )
    entries = [
        ManifestEntry(tmp_path / "known_a.jpg", "A", split="query"),
        ManifestEntry(tmp_path / "known_b.jpg", "B", split="query"),
        ManifestEntry(tmp_path / "unknown.jpg", "X", split="unknown"),
    ]

    metrics, predictions, pairs = evaluator.evaluate(entries)

    assert metrics["retrieval"]["recall_at_1"] == 1.0
    assert metrics["verification"]["far"] == 0.0
    assert metrics["verification"]["frr"] == 0.0
    assert metrics["verification"]["unknown_rejection_rate"] == 1.0

    evaluator.write_reports(
        metrics,
        predictions,
        pairs,
        metrics_path=tmp_path / "metrics.json",
        predictions_path=tmp_path / "predictions.csv",
        difficult_pairs_path=tmp_path / "pairs.csv",
        experiment={"name": "test"},
    )
    written = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert written["experiment"]["name"] == "test"
    assert (tmp_path / "predictions.csv").is_file()

