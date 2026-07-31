"""
Image Converter

Author: Ishita Dutta
Project: Universal Image Forensic Analysis Framework (UIFAF)

Converts a PIL image into a standardized
RGB NumPy array.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .exceptions import UnsupportedColorModeError


class ImageConverter:
    """
    Converts images into the standard
    RGB NumPy format used by the pipeline.
    """

    @staticmethod
    def to_rgb(image: Image.Image) -> np.ndarray:
        """
        Convert any supported image into RGB.

        Parameters
        ----------
        image : PIL.Image.Image

        Returns
        -------
        np.ndarray
            RGB Image
            Shape : (H,W,3)
            dtype : uint8
        """

        mode = image.mode

        # ---------------------------------------
        # RGB
        # ---------------------------------------

        if mode == "RGB":

            rgb = image

        # ---------------------------------------
        # RGBA
        # ---------------------------------------

        elif mode == "RGBA":

            rgb = image.convert("RGB")

        # ---------------------------------------
        # Grayscale
        # ---------------------------------------

        elif mode in ("L", "LA"):

            rgb = image.convert("RGB")

        # ---------------------------------------
        # Binary Image
        # ---------------------------------------

        elif mode == "1":

            rgb = image.convert("RGB")

        # ---------------------------------------
        # Palette Image
        # ---------------------------------------

        elif mode == "P":

            rgb = image.convert("RGB")

        # ---------------------------------------
        # CMYK
        # ---------------------------------------

        elif mode == "CMYK":

            rgb = image.convert("RGB")

        # ---------------------------------------
        # 16-bit
        # ---------------------------------------

        elif mode in ("I;16", "I", "F"):

            array = np.array(image)

            array = array.astype(np.float32)

            minimum = array.min()

            maximum = array.max()

            if maximum > minimum:

                array = (array - minimum) / (maximum - minimum)

            array = (array * 255).astype(np.uint8)

            rgb = Image.fromarray(array).convert("RGB")

        # ---------------------------------------
        # Unsupported
        # ---------------------------------------

        else:

            raise UnsupportedColorModeError(
                f"Unsupported image mode: {mode}"
            )

        rgb_array = np.asarray(rgb)

        return rgb_array