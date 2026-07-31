"""
Custom exceptions for the Backbone module.

This module defines all backbone-related exceptions used throughout
the ConvNeXt V2 feature extraction pipeline.
"""


class BackboneError(Exception):
    """Base exception for all backbone-related errors."""

    pass


class BackboneInitializationError(BackboneError):
    """
    Raised when the backbone model cannot be initialized.
    """

    pass


class InvalidFeatureMapError(BackboneError):
    """
    Raised when an extracted feature map has an unexpected
    shape, dtype, or channel configuration.
    """

    pass


class InvalidBackboneOutputError(BackboneError):
    """
    Raised when the backbone output is inconsistent or incomplete.
    """

    pass


class PretrainedWeightError(BackboneError):
    """
    Raised when pretrained weights cannot be loaded.
    """

    pass