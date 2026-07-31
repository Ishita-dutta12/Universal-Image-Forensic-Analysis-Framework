"""
UIFAF Visualization Exceptions

This module defines all custom exceptions used by the
Visualization & Localization module.
"""

from __future__ import annotations


class VisualizationError(Exception):
    """
    Base exception for the visualization module.
    """

    def __init__(self, message: str = "Visualization module error.") -> None:
        super().__init__(message)


class InvalidModelError(VisualizationError):
    """
    Raised when an invalid or unsupported model is supplied.
    """

    def __init__(self, message: str = "Invalid model supplied.") -> None:
        super().__init__(message)


class InvalidTargetLayerError(VisualizationError):
    """
    Raised when the Grad-CAM target layer cannot be located.
    """

    def __init__(self, message: str = "Invalid Grad-CAM target layer.") -> None:
        super().__init__(message)


class HeatmapGenerationError(VisualizationError):
    """
    Raised when heatmap generation fails.
    """

    def __init__(self, message: str = "Failed to generate heatmap.") -> None:
        super().__init__(message)


class OverlayGenerationError(VisualizationError):
    """
    Raised when overlay creation fails.
    """

    def __init__(self, message: str = "Failed to create overlay image.") -> None:
        super().__init__(message)


class LocalizationError(VisualizationError):
    """
    Raised when localization fails.
    """

    def __init__(self, message: str = "Localization failed.") -> None:
        super().__init__(message)


class VisualizationValidationError(VisualizationError):
    """
    Raised when visualization inputs fail validation.
    """

    def __init__(self, message: str = "Visualization validation failed.") -> None:
        super().__init__(message)