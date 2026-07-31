"""
Pixel Decoder

Author: Ishita Dutta

Universal Image Forensic Analysis Framework
"""

from pathlib import Path

import numpy as np

from PIL import ExifTags

from preprocessing.constants import BIT_DEPTH_MAPPING
from preprocessing.decoded_image import DecodedImage
from preprocessing.image_converter import ImageConverter
from preprocessing.image_info import ImageInfo
from preprocessing.image_loader import ImageLoader


class PixelDecoder:
    """
    Main Pixel Decoder.

    This class orchestrates the complete
    image decoding pipeline.
    """
    @staticmethod
    def _normalize_metadata(value):
        if isinstance(value, bytes):
            return value.hex()

        if hasattr(value, "numerator") and hasattr(value, "denominator"):
            return float(value)

        return value

    def decode(self, image_path: str | Path) -> DecodedImage:

        image_path = Path(image_path)

        # ----------------------------
        # Load Image
        # ----------------------------

        pil_image = ImageLoader.load(image_path)

        # ----------------------------
        # Original EXIF
        # ----------------------------

       # ----------------------------
# Original EXIF
# ----------------------------

        has_exif = False

        orientation = None

        raw_metadata = {}

        exif = pil_image.getexif()

        if exif:

            has_exif = True

            orientation = exif.get(
                ExifTags.Base.Orientation,
                None,
            )

    # Convert numeric EXIF tags into readable names
            for tag_id, value in exif.items():

                tag_name = ExifTags.TAGS.get(
                tag_id,
                tag_id,
            )

                raw_metadata[tag_name] = self._normalize_metadata(value)
        
    # ----------------------------
# Preserve container metadata
# (PNG/WebP/etc.)
# ----------------------------

        for key, value in pil_image.info.items():

            raw_metadata.setdefault(
            key,
            value,
        )

        # ----------------------------
        # Convert Image
        # ----------------------------

        rgb = ImageConverter.to_rgb(
            pil_image
        )

        # ----------------------------
        # Channels
        # ----------------------------

        if rgb.ndim == 2:

            channels = 1

        else:

            channels = rgb.shape[2]

        # ----------------------------
        # Bit Depth
        # ----------------------------

       

        bit_depth = BIT_DEPTH_MAPPING.get(
        pil_image.mode,
         8
        )

        # ----------------------------
        # Image Info
        # ----------------------------

        info = ImageInfo(

            filename=image_path.name,

            image_format=pil_image.format
            if pil_image.format
            else "Unknown",

            width=pil_image.width,

            height=pil_image.height,

            channels=channels,

            color_mode=pil_image.mode,

            bit_depth=bit_depth,

            file_size=image_path.stat().st_size,

            has_exif=has_exif,

            exif_orientation=orientation,
        )

        # ----------------------------
        # Return Object
        # ----------------------------

        return DecodedImage(
            filepath=image_path,

            original_image=pil_image,

            rgb_image=rgb,

            info=info,

            metadata=raw_metadata,

            validation_passed=True
        )