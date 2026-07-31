"""
Metadata Module

Author: Ishita Dutta

Universal Image Forensic Analysis Framework (UIFAF)
"""

from .exceptions import (
    MetadataError,
    MetadataReadError,
    MetadataParseError,
    MetadataValidationError,
    UnsupportedMetadataError,
)

from .metadata_models import (
    GPSInfo,
    CameraInfo,
    CaptureInfo,
    MetadataInfo,
)

from .exif_reader import ExifReader
from .metadata_parser import MetadataParser
from .metadata_validator import MetadataValidator


__all__ = [
    # Exceptions
    "MetadataError",
    "MetadataReadError",
    "MetadataParseError",
    "MetadataValidationError",
    "UnsupportedMetadataError",

    # Models
    "GPSInfo",
    "CameraInfo",
    "CaptureInfo",
    "MetadataInfo",

    # Components
    "ExifReader",
    "MetadataParser",
    "MetadataValidator",
]