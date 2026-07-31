from PIL import Image

from metadata.exif_reader import ExifReader


def test_read_image_without_exif(tmp_path):
    """
    Images without EXIF should return an empty dictionary.
    """

    image_path = tmp_path / "test.jpg"

    Image.new("RGB", (100, 100), "red").save(image_path)

    metadata = ExifReader.read(image_path)

    assert isinstance(metadata, dict)
    assert metadata == {}