"""
Backbone Validator

Author: Ishita Dutta
Project: UIFAF
"""

from __future__ import annotations

import torch

from .backbone_models import BackboneOutput
from .exceptions import (
    InvalidBackboneOutputError,
    InvalidFeatureMapError,
)


class BackboneValidator:
    """
    Validates the output produced by the backbone feature extractor.
    """

    @staticmethod
    def validate(backbone_output: BackboneOutput) -> None:
        """
        Validate a BackboneOutput object.

        Raises
        ------
        InvalidBackboneOutputError
            If BackboneOutput is invalid.
        InvalidFeatureMapError
            If any feature map is invalid.
        """

        if not isinstance(backbone_output, BackboneOutput):
            raise InvalidBackboneOutputError(
                "Expected BackboneOutput object."
            )

        feature_maps = backbone_output.stages

        if len(feature_maps) != 4:
            raise InvalidBackboneOutputError(
                "Backbone must produce exactly four feature maps."
            )

        device = None

        for index, fmap in enumerate(feature_maps, start=1):

            fmap.validate()

            tensor = fmap.tensor

            if not isinstance(tensor, torch.Tensor):
                raise InvalidFeatureMapError(
                    f"Stage {index}: tensor must be a torch.Tensor."
                )

            if tensor.ndim != 4:
                raise InvalidFeatureMapError(
                    f"Stage {index}: expected a 4D tensor (B,C,H,W)."
                )

            batch, channels, height, width = tensor.shape

            if batch <= 0:
                raise InvalidFeatureMapError(
                    f"Stage {index}: batch size must be positive."
                )

            if channels <= 0:
                raise InvalidFeatureMapError(
                    f"Stage {index}: channels must be positive."
                )

            if height <= 0 or width <= 0:
                raise InvalidFeatureMapError(
                    f"Stage {index}: invalid spatial dimensions."
                )

            if torch.isnan(tensor).any():
                raise InvalidFeatureMapError(
                    f"Stage {index}: tensor contains NaN values."
                )

            if torch.isinf(tensor).any():
                raise InvalidFeatureMapError(
                    f"Stage {index}: tensor contains Inf values."
                )

            if device is None:
                device = tensor.device

            elif tensor.device != device:
                raise InvalidFeatureMapError(
                    "All feature maps must be on the same device."
                )

        width, height = backbone_output.original_size

        if width <= 0 or height <= 0:
            raise InvalidBackboneOutputError(
                "Original image size must be positive."
            )