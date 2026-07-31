import pytest

from metadata.metadata_models import (
    CameraInfo,
    CaptureInfo,
    GPSInfo,
    MetadataInfo,
)

from metadata.metadata_validator import MetadataValidator

from metadata.exceptions import MetadataValidationError