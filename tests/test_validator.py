from PIL import Image
import pytest

from preprocessing.exceptions import (
    InvalidImageError,
    UnsupportedFormatError,
)
from preprocessing.image_validator import ImageValidator


def test_validator_valid_image(tmp_path):
    img_path = tmp_path / "test.jpg"

    Image.new("RGB", (50, 50), color="white").save(img_path)

    assert ImageValidator.validate(img_path) == img_path


def test_validator_non_existent_file():
    with pytest.raises(InvalidImageError):
        ImageValidator.validate("non_existent_file.jpg")


def test_validator_unsupported_format(tmp_path):
    txt_file = tmp_path / "invalid.txt"

    txt_file.write_text("This is not an image.")

    with pytest.raises(UnsupportedFormatError):
        ImageValidator.validate(txt_file)