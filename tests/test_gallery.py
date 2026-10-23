import numpy as np

from product_retrieval.retrieval import Gallery, GalleryBuilder


def test_gallery_roundtrip_without_pickle(tmp_path):
    gallery = Gallery(
        embeddings=np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32),
        metadata=[
            {"image_path": "a.jpg", "product_code": "A"},
            {"image_path": "b.jpg", "product_code": "B"},
        ],
        model_info={"model_name": "fake"},
        created_at="2026-01-01T00:00:00+00:00",
    )
    path = tmp_path / "gallery.npz"

    GalleryBuilder.save(gallery, path)
    loaded = GalleryBuilder.load(path)

    assert loaded.codes == ["A", "B"]
    assert np.allclose(loaded.embeddings, [[1, 0], [0, 1]])
    assert loaded.model_info["model_name"] == "fake"

