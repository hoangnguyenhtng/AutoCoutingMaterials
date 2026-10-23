import numpy as np
import pytest

torch = pytest.importorskip("torch")

from product_retrieval.models import DinoV2Encoder


class FakeDino(torch.nn.Module):
    def forward_features(self, batch):
        batch_size = batch.shape[0]
        cls = torch.arange(1, 5, dtype=torch.float32).repeat(batch_size, 1)
        patches = torch.stack([cls, cls * 2], dim=1)
        return {"x_norm_clstoken": cls, "x_norm_patchtokens": patches}


@pytest.mark.parametrize("feature_type", ["cls", "patch_mean"])
def test_encoder_outputs_l2_normalized_vectors(tmp_path, feature_type):
    from PIL import Image

    image_path = tmp_path / "part.png"
    Image.new("RGB", (80, 40), "gray").save(image_path)
    encoder = DinoV2Encoder(
        model_name="dinov2_vits14", feature_type=feature_type, model=FakeDino(), device="cpu"
    )

    embedding = encoder.encode_image(image_path)

    assert embedding.shape == (4,)
    assert np.linalg.norm(embedding) == pytest.approx(1.0)

