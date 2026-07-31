"""
UFRE (Universal Forensic Reasoning Engine) Exceptions.

This module defines the exception hierarchy used throughout the UFRE
reasoning pipeline. These exceptions provide fine-grained error reporting
for evidence encoding, reasoning, fusion, confidence estimation, and
validation.

Author: UIFAF
"""

from __future__ import annotations


class UFREError(Exception):
    """Base exception for all UFRE-related errors."""


# ============================================================================
# Validation Errors
# ============================================================================

class ValidationError(UFREError):
    """Base class for validation-related errors."""


class InvalidUFREInputError(ValidationError):
    """Raised when the input RepresentationOutput is invalid."""


class InvalidUFREOutputError(ValidationError):
    """Raised when the produced UFREOutput is invalid."""


class ShapeMismatchError(ValidationError):
    """Raised when tensor shapes are incompatible."""

    def __init__(self, expected: tuple, received: tuple):
        super().__init__(
            f"Shape mismatch. Expected {expected}, received {received}."
        )


class ChannelMismatchError(ValidationError):
    """Raised when feature channel dimensions differ."""

    def __init__(self, expected: int, received: int):
        super().__init__(
            f"Channel mismatch. Expected {expected}, received {received}."
        )


class DeviceMismatchError(ValidationError):
    """Raised when tensors reside on different devices."""

    def __init__(self, expected: str, received: str):
        super().__init__(
            f"Device mismatch. Expected '{expected}', received '{received}'."
        )


class EvidenceMismatchError(ValidationError):
    """Raised when evidence objects cannot be combined."""


# ============================================================================
# Processing Errors
# ============================================================================

class ProcessingError(UFREError):
    """Base class for reasoning pipeline processing errors."""


class EvidenceEncodingError(ProcessingError):
    """Raised when evidence encoding fails."""


class ReasoningAttentionError(ProcessingError):
    """Raised when reasoning attention computation fails."""


class EvidenceFusionError(ProcessingError):
    """Raised when evidence fusion fails."""


class ConfidenceEstimationError(ProcessingError):
    """Raised when confidence estimation fails."""


class UncertaintyEstimationError(ProcessingError):
    """Raised when uncertainty estimation fails."""


class StageReasoningError(ProcessingError):
    """
    Raised when reasoning fails for a specific backbone stage.
    """

    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(f"[{stage}] {message}")


# ============================================================================
# Configuration Errors
# ============================================================================

class ConfigurationError(UFREError):
    """Base class for configuration-related errors."""


class ReasoningConfigurationError(ConfigurationError):
    """Raised when the reasoning engine is configured incorrectly."""
