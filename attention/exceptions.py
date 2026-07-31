
"""
Exceptions for the Attention module.

Author: Ishita Dutta
Project: UIFAF
"""


class AttentionError(Exception):
    """
    Base class for Attention errors.
    """


class AttentionInitializationError(
    AttentionError
):
    """
    Raised when attention module cannot initialize.
    """


class InvalidAttentionFeatureError(
    AttentionError
):
    """
    Raised when an attention feature map is invalid.
    """


class InvalidAttentionOutputError(
    AttentionError
):
    """
    Raised when the complete attention output is invalid.
    """