"""
Metadata Validator

Author: Ishita Dutta

Universal Image Forensic Analysis Framework (UIFAF)
"""

from datetime import datetime

from .exceptions import MetadataValidationError
from .metadata_models import MetadataInfo


class MetadataValidator:
    """
    Validates parsed metadata.

    This class validates only the consistency of metadata.
    It does not modify or repair values.
    """

    @staticmethod
    def validate(metadata: MetadataInfo) -> MetadataInfo:
        """
        Validate parsed metadata.

        Parameters
        ----------
        metadata : MetadataInfo

        Returns
        -------
        MetadataInfo
        """

        MetadataValidator._validate_orientation(
            metadata.capture.orientation
        )

        MetadataValidator._validate_gps(
            metadata.gps.latitude,
            metadata.gps.longitude,
        )

        MetadataValidator._validate_datetime(
            metadata.capture.datetime_original
        )

        MetadataValidator._validate_datetime(
            metadata.capture.datetime_modified
        )

        return metadata

    # ---------------------------------------------------------
    # Orientation
    # ---------------------------------------------------------

    @staticmethod
    def _validate_orientation(
        orientation: int | None,
    ) -> None:

        if orientation is None:
            return

        if orientation not in range(1, 9):
            raise MetadataValidationError(
                f"Invalid EXIF orientation: {orientation}"
            )

    # ---------------------------------------------------------
    # GPS
    # ---------------------------------------------------------

    @staticmethod
    def _validate_gps(
        latitude: float | None,
        longitude: float | None,
    ) -> None:

        if latitude is not None:

            if not (-90 <= latitude <= 90):

                raise MetadataValidationError(
                    f"Invalid latitude: {latitude}"
                )

        if longitude is not None:

            if not (-180 <= longitude <= 180):

                raise MetadataValidationError(
                    f"Invalid longitude: {longitude}"
                )

    # ---------------------------------------------------------
    # DateTime
    # ---------------------------------------------------------

    @staticmethod
    def _validate_datetime(
        value: str | None,
    ) -> None:

        if value is None:
            return

        try:

            datetime.strptime(
                value,
                "%Y:%m:%d %H:%M:%S",
            )

        except ValueError as exc:

            raise MetadataValidationError(
                f"Invalid datetime: {value}"
            ) from exc