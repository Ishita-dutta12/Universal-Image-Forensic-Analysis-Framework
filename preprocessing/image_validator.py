"""
Image Validation Module

Author: Ishita Dutta
Project: Universal Image Forensic Analysis Framework (UIFAF)

This module validates an image before it enters the
forensic pipeline.
"""

from pathlib import Path

from PIL import Image, UnidentifiedImageError

from utils.logger import get_logger
from .constants import (
    MAX_FILE_SIZE_BYTES,
    SUPPORTED_EXTENSIONS,
    SUPPORTED_MODES,
)
from .exceptions import (
    CorruptedImageError,
    InvalidImageError,
    ImageTooLargeError,
    UnsupportedColorModeError,
    UnsupportedFormatError,
)

logger = get_logger(__name__)


class ImageValidator:
    """
    Performs image validation before decoding.
    """

    @staticmethod
    def validate(image_path: str | Path) -> Path:
        """
        Validate an image.

        Parameters
        ----------
        image_path : str | Path
            Path to image.

        Returns
        -------
        Path
            Validated Path object.

        Raises
        ------
        InvalidImageError
        UnsupportedFormatError
        ImageTooLargeError
        CorruptedImageError
        UnsupportedColorModeError
        """
        image_path = Path(image_path)

        # ----------------------------------------------------
        # File Exists
        # ----------------------------------------------------
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            raise InvalidImageError(f"Image not found:\n{image_path}")

        # ----------------------------------------------------
        # File Extension
        # ----------------------------------------------------
        extension = image_path.suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            logger.warning(f"Unsupported format attempt: {extension}")
            raise UnsupportedFormatError(
                f"Unsupported image format: {extension}"
            )

        # ----------------------------------------------------
        # File Size
        # ----------------------------------------------------
        file_size = image_path.stat().st_size

        if file_size > MAX_FILE_SIZE_BYTES:
            max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
            logger.warning(f"File size {file_size} exceeds limit of {max_mb} MB")
            raise ImageTooLargeError(
                f"Image size exceeds {max_mb:.0f} MB."
            )

        # ----------------------------------------------------
        # Verify Image Integrity
        # ----------------------------------------------------
        try:
            with Image.open(image_path) as image:
                image.verify()
        except UnidentifiedImageError:
            logger.error(f"Unidentified/Corrupted file: {image_path}")
            raise CorruptedImageError("Pillow could not identify the image.")
        except Exception as e:
            logger.error(f"Corrupted image check failed: {e}")
            raise CorruptedImageError(str(e))

        # ----------------------------------------------------
        # Check Color Mode (Re-open required after verify)
        # ----------------------------------------------------
        try:
            with Image.open(image_path) as image:
                if image.mode not in SUPPORTED_MODES:
                    logger.warning(f"Unsupported color mode: {image.mode}")
                    raise UnsupportedColorModeError(
                        f"Unsupported color mode: {image.mode}"
                    )
        except UnsupportedColorModeError:
            raise  # Re-raise explicit error without wrapping as CorruptedImageError
        except Exception as e:
            logger.error(f"Color mode validation error: {e}")
            raise CorruptedImageError(str(e))

        logger.info(f"Validation successful: {image_path.name}")
        return image_path