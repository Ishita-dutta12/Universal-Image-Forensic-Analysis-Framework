"""
Metadata Module Exceptions

Author: Ishita Dutta

Universal Image Forensic Analysis Framework (UIFAF)
"""


class MetadataError(Exception):
    """
    Base exception for all metadata-related errors.
    """

    pass


class MetadataReadError(MetadataError):
    """
    Raised when metadata cannot be read from an image.
    """

    pass


class MetadataParseError(MetadataError):
    """
    Raised when metadata exists but cannot be parsed correctly.
    """

    pass


class MetadataValidationError(MetadataError):
    """
    Raised when parsed metadata fails validation.
    """

    pass


class UnsupportedMetadataError(MetadataError):
    """
    Raised when unsupported or unknown metadata is encountered.
    """

    pass