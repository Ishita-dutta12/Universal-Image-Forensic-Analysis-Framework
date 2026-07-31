"""
Image Loader

Author: Ishita Dutta
Project: Universal Image Forensic Analysis Framework (UIFAF)

Loads an image safely using Pillow.
Handles EXIF orientation.
"""

from pathlib import Path

from PIL import Image
from PIL import ImageOps

from .image_validator import ImageValidator


class ImageLoader:
    """
    Safely loads an image using Pillow.

    Responsibilities:
    - Validate image
    - Open image
    - Apply EXIF orientation
    - Return PIL Image
    """

    @staticmethod
    def load(image_path: str | Path) -> Image.Image:
        """
        Load an image.

        Parameters
        ----------
        image_path : str | Path

        Returns
        -------
        PIL.Image.Image
        """

        # -------------------------------------------------
        # Validate First
        # -------------------------------------------------

        image_path = ImageValidator.validate(image_path)

        # -------------------------------------------------
        # Open Image
        # -------------------------------------------------

        with Image.open(image_path) as img:

    # Preserve original format
            image_format = img.format

            image = ImageOps.exif_transpose(img.copy())

    # Restore format
            image.format = image_format

        # -------------------------------------------------
        # Correct Orientation
        # -------------------------------------------------

       

        return image