
"""
Custom exceptions for the Representation Module.

Author: Ishita Dutta
Project: UIFAF
"""

from __future__ import annotations


class RepresentationError(Exception):
    """
    Base exception for the Representation Module.
    """


class RepresentationInitializationError(
    RepresentationError,
):
    """
    Raised when a representation encoder
    cannot be initialized.
    """


class InvalidRepresentationFeatureError(
    RepresentationError,
):
    """
    Raised when a representation feature
    is invalid.
    """


class InvalidRepresentationOutputError(
    RepresentationError,
):
    """
    Raised when the complete representation
    output is invalid.
    """