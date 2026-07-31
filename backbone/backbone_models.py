"""
Data models for the Backbone module.

These dataclasses provide strongly typed containers for the multi-scale
feature maps produced by the ConvNeXt V2 backbone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .exceptions import InvalidFeatureMapError


# ---------------------------------------------------------
# Individual Feature Map
# ---------------------------------------------------------

@dataclass(frozen=True)
class FeatureMap:
    """
    Represents one feature map extracted from a backbone stage.

    Attributes
    ----------
    stage : str
        Name of the backbone stage.

    tensor : torch.Tensor
        Feature tensor.

    channels : int
        Number of output channels.

    stride : int
        Effective stride relative to input image.
    """

    stage: str
    tensor: torch.Tensor
    channels: int
    stride: int

    def validate(self) -> None:
        """
        Validate feature map integrity.
        """

        if not isinstance(self.tensor, torch.Tensor):
            raise InvalidFeatureMapError(
                f"{self.stage}: tensor must be torch.Tensor."
            )

        if self.tensor.ndim != 4:
            raise InvalidFeatureMapError(
                f"{self.stage}: expected 4D tensor "
                f"(B,C,H,W), got {self.tensor.shape}"
            )

        if self.tensor.shape[1] != self.channels:
            raise InvalidFeatureMapError(
                f"{self.stage}: expected {self.channels} channels "
                f"but got {self.tensor.shape[1]}"
            )

        if self.stride <= 0:
            raise InvalidFeatureMapError(
                f"{self.stage}: stride must be positive."
            )

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns tensor shape.
        """
        return tuple(self.tensor.shape)

    @property
    def height(self) -> int:
        return self.tensor.shape[2]

    @property
    def width(self) -> int:
        return self.tensor.shape[3]


# ---------------------------------------------------------
# Backbone Output
# ---------------------------------------------------------

@dataclass
class BackboneOutput:
    """
    Complete output from ConvNeXt V2.

    Contains feature maps from every stage.

    stage1 -> high resolution

    stage4 -> deepest semantic features
    """

    stage1: FeatureMap
    stage2: FeatureMap
    stage3: FeatureMap
    stage4: FeatureMap

    original_size: Tuple[int, int]

    def validate(self) -> None:
        """
        Validate every feature map.
        """

        self.stage1.validate()
        self.stage2.validate()
        self.stage3.validate()
        self.stage4.validate()

        h, w = self.original_size

        if h <= 0 or w <= 0:
            raise InvalidFeatureMapError(
                "Invalid original image size."
            )

    @property
    def stages(self):
        """
        Returns feature maps in order.
        """
        return (
            self.stage1,
            self.stage2,
            self.stage3,
            self.stage4,
        )

    @property
    def num_stages(self) -> int:
        return 4

    @property
    def stage_names(self):
        return (
            self.stage1.stage,
            self.stage2.stage,
            self.stage3.stage,
            self.stage4.stage,
        )