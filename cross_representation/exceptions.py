"""
UIFAF - Cross Representation Interaction Exceptions

This module defines all custom exceptions used by the Cross
Representation Interaction (CRI) module.

Author: UIFAF
"""

from __future__ import annotations


class CrossRepresentationError(Exception):
    """
    Base exception for all Cross Representation Interaction errors.
    """

    def __init__(self, message: str = "Cross representation interaction error.") -> None:
        super().__init__(message)


class InvalidInteractionInputError(CrossRepresentationError):
    """
    Raised when the input RepresentationOutput is invalid.
    """

    def __init__(self, message: str = "Invalid interaction input.") -> None:
        super().__init__(message)


class InvalidInteractionOutputError(CrossRepresentationError):
    """
    Raised when the interaction output is invalid.
    """

    def __init__(self, message: str = "Invalid interaction output.") -> None:
        super().__init__(message)


class CrossAttentionError(CrossRepresentationError):
    """
    Raised when cross-attention computation fails.
    """

    def __init__(self, message: str = "Cross-attention computation failed.") -> None:
        super().__init__(message)


class GatedInteractionError(CrossRepresentationError):
    """
    Raised when gated interaction fails.
    """

    def __init__(self, message: str = "Gated interaction failed.") -> None:
        super().__init__(message)


class InteractionFusionError(CrossRepresentationError):
    """
    Raised when interaction fusion fails.
    """

    def __init__(self, message: str = "Interaction fusion failed.") -> None:
        super().__init__(message)


class InteractionConfigurationError(CrossRepresentationError):
    """
    Raised when an interaction module has an invalid configuration.
    """

    def __init__(self, message: str = "Invalid interaction configuration.") -> None:
        super().__init__(message)


class StageInteractionError(CrossRepresentationError):
    """
    Raised when processing a representation stage fails.
    """

    def __init__(self, stage: str, message: str | None = None) -> None:
        if message is None:
            message = f"Interaction failed for stage '{stage}'."
        super().__init__(message)
        self.stage = stage


class RepresentationMismatchError(CrossRepresentationError):
    """
    Raised when representation tensors are incompatible.
    """

    def __init__(
        self,
        expected: str,
        received: str,
    ) -> None:
        message = (
            f"Representation mismatch. "
            f"Expected '{expected}', received '{received}'."
        )
        super().__init__(message)
        self.expected = expected
        self.received = received


class ShapeMismatchError(CrossRepresentationError):
    """
    Raised when tensors have incompatible shapes.
    """

    def __init__(
        self,
        expected_shape: tuple[int, ...],
        received_shape: tuple[int, ...],
    ) -> None:
        message = (
            "Tensor shape mismatch. "
            f"Expected {expected_shape}, "
            f"received {received_shape}."
        )
        super().__init__(message)
        self.expected_shape = expected_shape
        self.received_shape = received_shape


class ChannelMismatchError(CrossRepresentationError):
    """
    Raised when channel dimensions do not match.
    """

    def __init__(
        self,
        expected_channels: int,
        received_channels: int,
    ) -> None:
        message = (
            f"Channel mismatch. Expected "
            f"{expected_channels}, received {received_channels}."
        )
        super().__init__(message)
        self.expected_channels = expected_channels
        self.received_channels = received_channels


class DeviceMismatchError(CrossRepresentationError):
    """
    Raised when tensors are located on different devices.
    """

    def __init__(
        self,
        expected_device: str,
        received_device: str,
    ) -> None:
        message = (
            f"Device mismatch. "
            f"Expected '{expected_device}', "
            f"received '{received_device}'."
        )
        super().__init__(message)
        self.expected_device = expected_device
        self.received_device = received_device
