"""
Validation utilities for the Attention module.

Author: Ishita Dutta
Project: UIFAF
"""

from __future__ import annotations

import torch

from .attention_models import AttentionOutput
from .exceptions import InvalidAttentionOutputError


class AttentionValidator:
    """
    Performs project-level validation on an AttentionOutput.
    """

    EXPECTED_STAGES = (
        "stage1",
        "stage2",
        "stage3",
        "stage4",
    )

    @staticmethod
    def validate(
        output: AttentionOutput,
    ) -> None:
        """
        Validate an AttentionOutput instance.
        """

        if not isinstance(output, AttentionOutput):
            raise InvalidAttentionOutputError(
                "Expected AttentionOutput."
            )

        output.validate()

        stages = output.stages

        if len(stages) != 4:
            raise InvalidAttentionOutputError(
                "Exactly four attention stages are required."
            )

        device = stages[0].tensor.device

        for expected_name, feature in zip(
            AttentionValidator.EXPECTED_STAGES,
            stages,
        ):

            if feature.stage != expected_name:
                raise InvalidAttentionOutputError(
                    f"Expected {expected_name}, got {feature.stage}."
                )

            if not torch.isfinite(feature.tensor).all():
                raise InvalidAttentionOutputError(
                    f"{feature.stage}: tensor contains NaN or Inf."
                )

            if feature.tensor.device != device:
                raise InvalidAttentionOutputError(
                    "All tensors must be on the same device."
                )

            if feature.attention_map is None:
                raise InvalidAttentionOutputError(
                    f"{feature.stage}: missing attention map."
                )

            if not torch.isfinite(feature.attention_map).all():
                raise InvalidAttentionOutputError(
                    f"{feature.stage}: attention map contains NaN or Inf."
                )

            if feature.attention_map.device != device:
                raise InvalidAttentionOutputError(
                    "Attention maps must be on the same device."
                )

            if feature.attention_map.shape[0] != feature.tensor.shape[0]:
                raise InvalidAttentionOutputError(
                    f"{feature.stage}: batch size mismatch."
                )

            if feature.attention_map.shape[-2:] != feature.tensor.shape[-2:]:
                raise InvalidAttentionOutputError(
                    f"{feature.stage}: spatial size mismatch."
                )

        h, w = output.original_size

        if h <= 0 or w <= 0:
            raise InvalidAttentionOutputError(
                "Invalid original image size."
            )
