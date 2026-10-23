"""Retrieval and verification evaluation with CSV/JSON reporting."""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

from product_retrieval.data import ManifestEntry
from product_retrieval.verification import ProductVerifier, VerificationThresholds


class Evaluator:
    def __init__(self, encoder, retriever, thresholds: VerificationThresholds | Mapping[str, float]):
        self.encoder = encoder
        self.retriever = retriever
        self.verifier = ProductVerifier(retriever)
        self.thresholds = (
            thresholds
            if isinstance(thresholds, VerificationThresholds)
            else VerificationThresholds.from_mapping(thresholds)
        )

    def evaluate(self, entries: Iterable[ManifestEntry], batch_size: int = 16) -> tuple[dict, list[dict], Counter]:
        rows = list(entries)
        if not rows:
            raise ValueError("No query/unknown entries supplied for evaluation")
        started = time.perf_counter()
        embeddings = self.encoder.encode_batch([row.image_path for row in rows], batch_size=batch_size)
        embedding_seconds = time.perf_counter() - started

        predictions: list[dict] = []
        difficult_pairs: Counter = Counter()
        known_count = recall1 = recall5 = 0
        reciprocal_rank_sum = 0.0
        false_accepts = false_rejects = positive_count = negative_count = 0
        reviews = unknown_count = unknown_rejections = 0
        query_seconds = 0.0

        for entry, embedding in zip(rows, embeddings):
            query_started = time.perf_counter()
            ranking = self.retriever.rank_codes(embedding, top_k=max(5, len(self.retriever.gallery.codes)))
            is_known = entry.product_code in self.retriever.gallery.codes and entry.split != "unknown"
            predicted = ranking[0]["product_code"] if ranking else None
            rank = next(
                (item["rank"] for item in ranking if item["product_code"] == entry.product_code), None
            )

            if is_known:
                known_count += 1
                recall1 += int(rank == 1)
                recall5 += int(rank is not None and rank <= 5)
                reciprocal_rank_sum += 0.0 if rank is None else 1.0 / rank
                if predicted != entry.product_code and predicted is not None:
                    difficult_pairs[(entry.product_code, predicted)] += 1

                positive = self.verifier.verify(embedding, entry.product_code, self.thresholds)
                positive_count += 1
                false_rejects += int(positive["status"] != "MATCH")
                reviews += int(positive["status"] == "REVIEW")

                wrong_code = next(
                    (item["product_code"] for item in ranking if item["product_code"] != entry.product_code),
                    None,
                )
                if wrong_code is not None:
                    negative = self.verifier.verify(embedding, wrong_code, self.thresholds)
                    negative_count += 1
                    false_accepts += int(negative["status"] == "MATCH")
            else:
                unknown_count += 1
                if predicted is None:
                    verification = {"status": "UNKNOWN"}
                else:
                    verification = self.verifier.verify(embedding, predicted, self.thresholds)
                unknown_rejections += int(verification["status"] != "MATCH")
                reviews += int(verification["status"] == "REVIEW")
                positive = verification

            query_seconds += time.perf_counter() - query_started
            predictions.append(
                {
                    "image_path": str(entry.image_path),
                    "split": entry.split,
                    "true_code": entry.product_code,
                    "predicted_code": predicted or "",
                    "true_code_rank": rank or "",
                    "status": positive["status"],
                    "top1_score": ranking[0]["score"] if ranking else "",
                }
            )

        metrics = {
            "counts": {"total": len(rows), "known": known_count, "unknown": unknown_count},
            "retrieval": {
                "recall_at_1": recall1 / known_count if known_count else None,
                "recall_at_5": recall5 / known_count if known_count else None,
                "mrr": reciprocal_rank_sum / known_count if known_count else None,
            },
            "verification": {
                "far": false_accepts / negative_count if negative_count else None,
                "frr": false_rejects / positive_count if positive_count else None,
                "review_rate": reviews / len(rows),
                "unknown_rejection_rate": unknown_rejections / unknown_count if unknown_count else None,
            },
            "timing_seconds": {
                "embedding_total": embedding_seconds,
                "embedding_per_image": embedding_seconds / len(rows),
                "query_total": query_seconds,
                "query_per_image": query_seconds / len(rows),
            },
        }
        return metrics, predictions, difficult_pairs

    @staticmethod
    def write_reports(
        metrics: dict,
        predictions: list[dict],
        difficult_pairs: Counter,
        *,
        metrics_path: str | Path,
        predictions_path: str | Path,
        difficult_pairs_path: str | Path,
        experiment: dict | None = None,
    ) -> None:
        metrics_output = Path(metrics_path)
        predictions_output = Path(predictions_path)
        pairs_output = Path(difficult_pairs_path)
        for output in (metrics_output, predictions_output, pairs_output):
            output.parent.mkdir(parents=True, exist_ok=True)

        payload = {"experiment": experiment or {}, **metrics}
        metrics_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        prediction_fields = [
            "image_path", "split", "true_code", "predicted_code", "true_code_rank", "status", "top1_score"
        ]
        with predictions_output.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=prediction_fields)
            writer.writeheader()
            writer.writerows(predictions)
        with pairs_output.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["true_code", "predicted_code", "count"])
            for (true_code, predicted_code), count in difficult_pairs.most_common():
                writer.writerow([true_code, predicted_code, count])
