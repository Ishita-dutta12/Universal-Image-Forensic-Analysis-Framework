"""
Cross Representation Gated Interaction.

This module adaptively refines each representation using
lightweight learnable gates.

Rather than exchanging information through attention,
each representation learns how much of its own information
should be retained.

Pipeline
--------
RepresentationOutput
        │
        ▼
GatedInteraction
        │
        ▼
Stage-wise Gating
        │
        ▼
Enhanced RepresentationOutput

Author: Ishita Dutta
Project: UIFAF
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from representation.representation_models import (
    RepresentationFeature,
    StageRepresentation,
)

from .base_interaction import BaseInteraction
from .exceptions import (
    GatedInteractionError,
)
from .interaction_validator import InteractionValidator


class GatedInteractionBlock(nn.Module):
    """
    Lightweight gated refinement block.

    Each available representation is refined independently
    using a learnable sigmoid gate while preserving spatial
    dimensions.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(
                "channels must be positive."
            )

        if reduction <= 0:
            raise ValueError(
                "reduction must be positive."
            )

        hidden_channels = max(
            channels // reduction,
            16,
        )

        # -----------------------------------------
        # Shared Gate Network
        # -----------------------------------------

        self.gate = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.Sigmoid(),
        )

        # -----------------------------------------
        # Refinement Projection
        # -----------------------------------------

        self.refinement = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )

        # -----------------------------------------
        # LayerScale
        # -----------------------------------------

        self.layer_scale = nn.Parameter(
            torch.ones(channels) * 1e-4
        )
        
    def _refine_feature(
        self,
        feature: Optional[RepresentationFeature],
    ) -> Optional[RepresentationFeature]:
        """
        Refine a single representation feature using
        adaptive sigmoid gating.
        """
        if feature is None:
            return None

        x = feature.tensor

        BaseInteraction.ensure_finite(x)

        residual = x

        gate = self.gate(x)

        refined = self.refinement(x * gate)

        refined = (
            refined * self.layer_scale.view(
                1,
                -1,
                1,
                1,
            )
            + residual
        )

        BaseInteraction.ensure_finite(refined)

        return BaseInteraction.clone_feature(
            feature,
            refined,
        )

    def forward(
        self,
        spatial: Optional[RepresentationFeature],
        frequency: Optional[RepresentationFeature],
        context: Optional[RepresentationFeature],
        fused: Optional[RepresentationFeature],
    ) -> tuple[
        Optional[RepresentationFeature],
        Optional[RepresentationFeature],
        Optional[RepresentationFeature],
        Optional[RepresentationFeature],
    ]:
        """
        Apply gated refinement independently to every
        available representation.
        """

        spatial = self._refine_feature(spatial)

        frequency = self._refine_feature(frequency)

        context = self._refine_feature(context)

        fused = self._refine_feature(fused)

        return (
            spatial,
            frequency,
            context,
            fused,
        )
        
class GatedInteraction(BaseInteraction):
    """
    Stage-wise gated representation interaction.

    Each stage is processed independently using a
    lightweight gating mechanism that adaptively
    refines every available representation.
    """

    def __init__(
        self,
        stage_channels: tuple[int, ...] = (
            96,
            192,
            384,
            768,
        ),
        reduction: int = 4,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        if len(stage_channels) != 4:
            raise GatedInteractionError(
                "Exactly four stage channel sizes are required."
            )

        self.blocks = nn.ModuleDict(
            {
                f"stage{i + 1}": GatedInteractionBlock(
                    channels=channels,
                    reduction=reduction,
                    dropout=dropout,
                )
                for i, channels in enumerate(stage_channels)
            }
        )

    def interact_stage(
        self,
        stage: StageRepresentation,
    ) -> StageRepresentation:
        """
        Refine a single stage using gated interaction.
        """

        InteractionValidator.validate_shape_consistency(stage)
        InteractionValidator.validate_device_consistency(stage)
        InteractionValidator.validate_dtype_consistency(stage)
        InteractionValidator.validate_channel_consistency(stage)
        InteractionValidator.validate_batch_consistency(stage)
        InteractionValidator.validate_finite(stage)

        if stage.stage not in self.blocks:
            raise GatedInteractionError(
                f"Unknown stage '{stage.stage}'."
            )

        block = self.blocks[stage.stage]

        (
            spatial,
            frequency,
            context,
            fused,
        ) = block(
            stage.spatial,
            stage.frequency,
            stage.context,
            stage.fused,
        )

        output = StageRepresentation(
            stage=stage.stage,
            spatial=spatial,
            frequency=frequency,
            context=context,
            fused=fused,
        )

        output.validate()

        return output
    def extra_repr(self) -> str:
        """
        Extra representation for debugging and model summary.
        """
        return (
            f"stages={list(self.blocks.keys())}, "
            f"num_blocks={len(self.blocks)}"
        )