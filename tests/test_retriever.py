import numpy as np
import pytest

from product_retrieval.retrieval import Gallery, ProductRetriever


def make_gallery():
    return Gallery(
        embeddings=np.array(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32
        ),
        metadata=[
            {"image_path": "a1", "product_code": "A"},
            {"image_path": "a2", "product_code": "A"},
            {"image_path": "b1", "product_code": "B"},
            {"image_path": "b2", "product_code": "B"},
        ],
        model_info={},
        created_at="now",
    )


@pytest.mark.parametrize("aggregation", ["top_mean", "max", "centroid"])
def test_rank_codes(aggregation):
    retriever = ProductRetriever(make_gallery(), aggregation=aggregation, references_per_code=2)

    ranked = retriever.rank_codes(np.array([1.0, 0.0]), top_k=2)

    assert [row["product_code"] for row in ranked] == ["A", "B"]
    assert ranked[0]["rank"] == 1


def test_search_returns_reference_metadata():
    retriever = ProductRetriever(make_gallery())
    result = retriever.search([0.0, 1.0], top_k=1)
    assert result[0]["product_code"] == "B"
    assert result[0]["image_path"] == "b1"

