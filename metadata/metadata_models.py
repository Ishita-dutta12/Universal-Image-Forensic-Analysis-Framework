"""
Metadata Models

Author: Ishita Dutta

Universal Image Forensic Analysis Framework (UIFAF)
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class GPSInfo:
    """
    GPS metadata extracted from the image.
    """

    latitude: float | None = None
    longitude: float | None = None
    altitude: float | None = None


@dataclass(slots=True, frozen=True)
class CameraInfo:
    """
    Camera and software information.
    """

    make: str | None = None
    model: str | None = None
    lens_model: str | None = None
    software: str | None = None


@dataclass(slots=True, frozen=True)
class CaptureInfo:
    """
    Capture-related metadata.
    """

    datetime_original: str | None = None
    datetime_modified: str | None = None
    orientation: int | None = None


@dataclass(slots=True)
class MetadataInfo:
    """
    Complete structured metadata extracted from an image.
    """

    camera: CameraInfo = field(default_factory=CameraInfo)

    gps: GPSInfo = field(default_factory=GPSInfo)

    capture: CaptureInfo = field(default_factory=CaptureInfo)

    raw_exif: dict[str, Any] = field(default_factory=dict)