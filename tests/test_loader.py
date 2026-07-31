from PIL import Image

from preprocessing.image_loader import ImageLoader


def test_loader_opens_valid_image(tmp_path):
    img_path = tmp_path / "test.jpg"

    Image.new("RGB", (100, 100), color="green").save(img_path)

    pil_img = ImageLoader.load(img_path)

    assert isinstance(pil_img, Image.Image)
    assert pil_img.size == (100, 100)
    assert pil_img.mode == "RGB"