"""
EXIF Reader

Author: Ishita Dutta

Universal Image Forensic Analysis Framework (UIFAF)
"""

from pathlib import Path
from typing import Any

from PIL import Image
from PIL.ExifTags import TAGS

from .exceptions import MetadataReadError


class ExifReader:
    """
    Reads raw EXIF metadata from an image.

    Responsibilities
    ----------------
    - Read EXIF metadata
    - Decode EXIF tag IDs into readable names
    - Preserve raw values
    - Return a dictionary

    This class DOES NOT:
    - Parse GPS
    - Validate metadata
    - Normalize values
    """

    @staticmethod
    def read(image_path: str | Path) -> dict[str, Any]:
        """
        Read raw EXIF metadata.

        Parameters
        ----------
        image_path : str | Path

        Returns
        -------
        dict[str, Any]
            Dictionary containing decoded EXIF tags.
        """

        image_path = Path(image_path)

        try:
            with Image.open(image_path) as image:

                exif = image.getexif()

                if not exif:
                    return {}

                metadata = {}

                for tag_id, value in exif.items():

                    tag_name = TAGS.get(tag_id, str(tag_id))

                    metadata[tag_name] = value

                return metadata

        except Exception as exc:
            raise MetadataReadError(
                f"Unable to read metadata from '{image_path}'."
            ) from exc