"""
Spatial Representation Encoder.

Learns forensic spatial representations from enhanced
attention feature maps.

Author: Ishita Dutta
Project: UIFAF
"""

from __future__ import annotations

import torch
import torch.nn as nn

from attention.attention_models import (
    AttentionFeatureMap,
    AttentionOutput,
)

from .exceptions import (
    RepresentationInitializationError,
)

from .representation_models import (
    RepresentationFeature,
    RepresentationOutput,
    StageRepresentation,
)


class SpatialBlock(nn.Module):
    """
    Lightweight residual spatial encoder block.
    """

    def __init__(self, channels: int):

        super().__init__()

        if channels <= 0:
            raise RepresentationInitializationError(
                "channels must be positive."
            )

        groups = min(32, channels)

        while channels % groups != 0:
            groups -= 1

        self.block = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(
                groups,
                channels,
            ),

            nn.GELU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return x + self.block(x)


class SpatialEncoder(nn.Module):
    """
    Produces spatial forensic representations.
    """

    def __init__(
        self,
        stage_channels: dict[str, int],
    ):

        super().__init__()

        try:

            self.stage1 = SpatialBlock(
                stage_channels["stage1"]
            )

            self.stage2 = SpatialBlock(
                stage_channels["stage2"]
            )

            self.stage3 = SpatialBlock(
                stage_channels["stage3"]
            )

            self.stage4 = SpatialBlock(
                stage_channels["stage4"]
            )

        except Exception as exc:

            raise RepresentationInitializationError(
                str(exc)
            ) from exc

    def _encode_stage(
        self,
        feature: AttentionFeatureMap,
        encoder: SpatialBlock,
    ) -> RepresentationFeature:

        tensor = encoder(feature.tensor)

        return RepresentationFeature(
            stage=feature.stage,
            tensor=tensor,
            representation_type="spatial",
        )

    def forward(
        self,
        attention_output: AttentionOutput,
    ) -> dict[str, StageRepresentation | tuple[int, int]]:

        attention_output.validate()

        return {
            "stage1": self._encode_stage(
                attention_output.stage1,
                self.stage1,
            ),
            "stage2": self._encode_stage(
                attention_output.stage2,
                self.stage2,
            ),
            "stage3": self._encode_stage(
                attention_output.stage3,
                self.stage3,
            ),
            "stage4": self._encode_stage(
                attention_output.stage4,
                self.stage4,
            ),
            "original_size": attention_output.original_size,
        }