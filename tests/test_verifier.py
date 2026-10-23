import numpy as np

from product_retrieval.retrieval import Gallery, ProductRetriever
from product_retrieval.verification import ProductVerifier, VerificationThresholds


def verifier():
    gallery = Gallery(
        embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        metadata=[
            {"image_path": "a", "product_code": "A"},
            {"image_path": "b", "product_code": "B"},
        ],
        model_info={},
        created_at="now",
    )
    return ProductVerifier(ProductRetriever(gallery, aggregation="max"))


THRESHOLDS = VerificationThresholds(
    match_score=0.8, mismatch_score=0.8, unknown_score=0.5, margin=0.1
)


def test_match():
    result = verifier().verify([1.0, 0.0], "A", THRESHOLDS)
    assert result["status"] == "MATCH"
    assert result["best_other_code"] == "B"


def test_mismatch():
    result = verifier().verify([0.0, 1.0], "A", THRESHOLDS)
    assert result["status"] == "MISMATCH"


def test_review_when_codes_are_close():
    result = verifier().verify([1.0, 1.0], "A", THRESHOLDS)
    assert result["status"] == "REVIEW"


def test_not_registered():
    result = verifier().verify([1.0, 0.0], "C", THRESHOLDS)
    assert result["status"] == "NOT_REGISTERED"

