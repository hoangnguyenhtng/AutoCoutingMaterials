"""Product-code verification and open-set rejection rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from product_retrieval.retrieval import ProductRetriever


@dataclass(frozen=True)
class VerificationThresholds:
    match_score: float
    mismatch_score: float
    unknown_score: float
    margin: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> "VerificationThresholds":
        return cls(
            match_score=float(values["match_score"]),
            mismatch_score=float(values.get("mismatch_score", values["match_score"])),
            unknown_score=float(values["unknown_score"]),
            margin=float(values["margin"]),
        )


class ProductVerifier:
    def __init__(self, retriever: ProductRetriever):
        self.retriever = retriever

    def verify(
        self,
        query_embedding,
        selected_code: str,
        thresholds: VerificationThresholds | Mapping[str, float],
    ) -> dict[str, object]:
        if not isinstance(thresholds, VerificationThresholds):
            thresholds = VerificationThresholds.from_mapping(thresholds)
        scores = self.retriever.all_code_scores(query_embedding)
        if selected_code not in scores:
            return {
                "status": "NOT_REGISTERED",
                "selected_code": selected_code,
                "target_score": None,
                "best_other_code": None,
                "best_other_score": None,
                "margin": None,
            }

        target_score = scores[selected_code]
        competitors = [(code, score) for code, score in scores.items() if code != selected_code]
        if competitors:
            best_other_code, best_other_score = max(competitors, key=lambda item: item[1])
        else:
            best_other_code, best_other_score = None, float("-inf")
        margin = target_score - best_other_score
        best_score = max(scores.values())

        if best_score < thresholds.unknown_score:
            status = "UNKNOWN"
        elif target_score >= thresholds.match_score and margin >= thresholds.margin:
            status = "MATCH"
        elif best_other_score >= thresholds.mismatch_score and margin <= -thresholds.margin:
            status = "MISMATCH"
        else:
            status = "REVIEW"

        return {
            "status": status,
            "selected_code": selected_code,
            "target_score": float(target_score),
            "best_other_code": best_other_code,
            "best_other_score": None if best_other_code is None else float(best_other_score),
            "margin": None if best_other_code is None else float(margin),
        }

