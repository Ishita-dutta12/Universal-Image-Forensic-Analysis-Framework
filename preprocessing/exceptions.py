"""
Custom Exceptions for UIFAF
"""


class ImageProcessingError(Exception):
    """
    Base exception for all image processing errors.
    """

    pass


class InvalidImageError(ImageProcessingError):
    """
    Image path does not exist.
    """

    pass


class UnsupportedFormatError(ImageProcessingError):
    """
    Unsupported extension.
    """

    pass


class CorruptedImageError(ImageProcessingError):
    """
    Corrupted image.
    """

    pass


class ImageTooLargeError(ImageProcessingError):
    """
    Image exceeds size limit.
    """

    pass


class UnsupportedColorModeError(ImageProcessingError):
    """
    Unsupported Pillow color mode.
    """

    pass