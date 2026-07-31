"""
Metadata Parser

Author: Ishita Dutta

Universal Image Forensic Analysis Framework (UIFAF)
"""

from typing import Any

from .exceptions import MetadataParseError
from .metadata_models import (
    CameraInfo,
    CaptureInfo,
    GPSInfo,
    MetadataInfo,
)


class MetadataParser:
    """
    Converts raw EXIF metadata into structured metadata objects.
    """

    @staticmethod
    def parse(raw_exif: dict[str, Any]) -> MetadataInfo:
        """
        Parse raw EXIF metadata into structured metadata.

        Parameters
        ----------
        raw_exif : dict[str, Any]

        Returns
        -------
        MetadataInfo
        """

        try:

            camera = CameraInfo(
                make=raw_exif.get("Make"),
                model=raw_exif.get("Model"),
                lens_model=raw_exif.get("LensModel"),
                software=raw_exif.get("Software"),
            )

            capture = CaptureInfo(
                datetime_original=raw_exif.get("DateTimeOriginal"),
                datetime_modified=raw_exif.get("DateTime"),
                orientation=raw_exif.get("Orientation"),
            )

            gps = GPSInfo()

            gps_info = raw_exif.get("GPSInfo")

            if gps_info:

                latitude = MetadataParser._parse_gps(
                    gps_info,
                    latitude=True,
                )

                longitude = MetadataParser._parse_gps(
                    gps_info,
                    latitude=False,
                )

                altitude = gps_info.get(6)

                if altitude is not None:
                    altitude = MetadataParser._to_float(altitude)

                gps = GPSInfo(
                    latitude=latitude,
                    longitude=longitude,
                    altitude=altitude,
                )

            return MetadataInfo(
                camera=camera,
                gps=gps,
                capture=capture,
                raw_exif=raw_exif,
            )

        except Exception as exc:
            raise MetadataParseError(
                "Failed to parse metadata."
            ) from exc

    @staticmethod
    def _to_float(value: Any) -> float:
        """
        Convert EXIF numeric values to float.

        Supports:
        - int
        - float
        - Pillow IFDRational
        - (numerator, denominator) tuples
        """

        if isinstance(value, (int, float)):
            return float(value)

        if hasattr(value, "numerator") and hasattr(value, "denominator"):

            if value.denominator == 0:
                raise ValueError("Division by zero.")

            return value.numerator / value.denominator

        if isinstance(value, tuple) and len(value) == 2:

            numerator, denominator = value

            if denominator == 0:
                raise ValueError("Division by zero.")

            return numerator / denominator

        raise TypeError(
            f"Unsupported EXIF numeric type: {type(value)}"
        )

    @staticmethod
    def _parse_gps(
        gps_info: dict,
        *,
        latitude: bool,
    ) -> float | None:
        """
        Convert EXIF GPS coordinates from
        Degrees / Minutes / Seconds
        to decimal degrees.
        """

        try:

            if latitude:

                values = gps_info.get(2)
                ref = gps_info.get(1)

            else:

                values = gps_info.get(4)
                ref = gps_info.get(3)

            if not values:
                return None

            degrees = MetadataParser._to_float(values[0])
            minutes = MetadataParser._to_float(values[1])
            seconds = MetadataParser._to_float(values[2])

            decimal = (
                degrees
                + minutes / 60
                + seconds / 3600
            )

            if ref in ("S", "W"):
                decimal *= -1

            return decimal

        except Exception:
            return None