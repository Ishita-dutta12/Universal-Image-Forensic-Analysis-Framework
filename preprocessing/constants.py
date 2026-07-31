"""
Shared constants.
"""

from configs import settings

SUPPORTED_EXTENSIONS = set(
    settings.image.supported_extensions
)

SUPPORTED_MODES = set(
    settings.image.supported_modes
)

MAX_FILE_SIZE_MB = (
    settings.image.max_file_size_mb
)

MAX_FILE_SIZE_BYTES = (
    MAX_FILE_SIZE_MB * 1024 * 1024
)

BIT_DEPTH_MAPPING = (
    settings.bit_depth_mapping
)