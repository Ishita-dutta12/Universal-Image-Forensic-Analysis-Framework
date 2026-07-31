import numpy as np
from PIL import Image

from preprocessing.image_converter import ImageConverter
from preprocessing.image_loader import ImageLoader


def test_converter_to_rgb(tmp_path):
    img_path = tmp_path / "test.jpg"

    Image.new("RGB", (100, 100), color="red").save(img_path)

    pil_img = ImageLoader.load(img_path)
    rgb_img = ImageConverter.to_rgb(pil_img)

    assert isinstance(rgb_img, np.ndarray)
    assert rgb_img.shape == (100, 100, 3)
    assert rgb_img.dtype == np.uint8