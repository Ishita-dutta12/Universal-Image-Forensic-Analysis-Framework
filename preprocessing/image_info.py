"""
Image Information

Author: Ishita Dutta
Project: UIFAF
"""

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ImageInfo:
    """
    Stores image information extracted during decoding.
    """

    filename: str

    image_format: str

    width: int

    height: int

    channels: int

    color_mode: str

    bit_depth: int

    file_size: int

    has_exif: bool

    exif_orientation: int | None
    
    @property
    def size(self) -> tuple[int, int]:
        """
        Returns image dimensions as (height, width).
        """
        return (self.height, self.width)