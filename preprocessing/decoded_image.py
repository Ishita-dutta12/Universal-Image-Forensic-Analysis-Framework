from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .image_info import ImageInfo




@dataclass
class DecodedImage:
    rgb_image: np.ndarray
    original_image: Image.Image
    info: ImageInfo
    filepath: Path
    validation_passed: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class PixelDecoder:
    def decode(self, filepath: Any) -> DecodedImage:
        filepath = Path(filepath)
        img = Image.open(filepath)
        original = img.copy()
        rgb_image = img.convert("RGB")
        rgb_array = np.array(rgb_image)
        info = ImageInfo(width=rgb_image.width, height=rgb_image.height)
        validation_passed = (
            rgb_array.dtype == np.uint8
            and rgb_array.shape == (rgb_image.height, rgb_image.width, 3)
        )
        return DecodedImage(
            rgb_image=rgb_array,
            original_image=original,
            info=info,
            filepath=filepath,
            validation_passed=validation_passed,
            metadata={},
        )


def test_pixel_decoder_full_pipeline(tmp_path):
    img_path = tmp_path / "test.jpg"

    Image.new("RGB", (64, 64), color="blue").save(img_path)

    decoder = PixelDecoder()
    decoded = decoder.decode(img_path)

    # RGB image
    assert isinstance(decoded.rgb_image, np.ndarray)
    assert decoded.rgb_image.shape == (64, 64, 3)
    assert decoded.rgb_image.dtype == np.uint8

    # Original image
    assert decoded.original_image.size == (64, 64)

    # Image informationa
    assert decoded.info.width == 64
    assert decoded.info.height == 64

    # File information
    assert decoded.filepath == img_path

    # Validation
    assert decoded.validation_passed is True