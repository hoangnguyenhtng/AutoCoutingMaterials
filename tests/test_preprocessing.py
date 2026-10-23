from PIL import Image

from product_retrieval.preprocessing import ImagePreprocessor


def test_letterbox_preserves_shape_and_adds_padding():
    image = Image.new("RGB", (200, 100), (255, 0, 0))
    preprocessor = ImagePreprocessor(image_size=224, background_color=(255, 255, 255))

    result = preprocessor.prepare_pil(image)

    assert result.size == (224, 224)
    assert result.getpixel((112, 0)) == (255, 255, 255)
    assert result.getpixel((112, 112))[0] > 240
    assert result.getpixel((112, 112))[1] < 15


def test_image_size_must_match_patch_grid():
    try:
        ImagePreprocessor(image_size=225)
    except ValueError as error:
        assert "multiple" in str(error)
    else:
        raise AssertionError("Expected ValueError")

