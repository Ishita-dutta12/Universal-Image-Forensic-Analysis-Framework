"""
Cross Representation Interaction Fusion.

This module fuses the outputs of different interaction
mechanisms into a unified stage representation.

Current supported interaction sources:

    • Cross Attention
    • Gated Interaction

Pipeline
--------
RepresentationOutput
        │
        ▼
Interaction Fusion
        │
        ▼
RepresentationOutput

Author: Ishita Dutta
Project: UIFAF
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from representation.representation_models import (
    RepresentationFeature,
    RepresentationOutput,
    StageRepresentation,
)
from tests.test_context_encoder import attention_output

from .base_interaction import BaseInteraction
from .exceptions import (
    InteractionFusionError,
)
from .interaction_validator import (
    InteractionValidator,
)
class InteractionFusionBlock(nn.Module):
    """
    Lightweight feature fusion block.

    Fuses two refined representations using
    learnable weighted residual fusion.
    """

    def __init__(
        self,
        channels: int,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(
                "channels must be positive."
            )

        self.projection = nn.Sequential(
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

        self.weight_a = nn.Parameter(
            torch.tensor(1.0)
        )

        self.weight_b = nn.Parameter(
            torch.tensor(1.0)
        )

        self.layer_scale = nn.Parameter(
            torch.ones(channels) * 1e-4
        )
        
    def _fuse_feature(
        self,
        first: Optional[RepresentationFeature],
        second: Optional[RepresentationFeature],
    ) -> Optional[RepresentationFeature]:
        

            if first is None and second is None:
                return None

            if first is None:
                return BaseInteraction.clone_feature(
                    second,
                    second.tensor,
                )

            if second is None:
                return BaseInteraction.clone_feature(
                    first,
                    first.tensor,
                )

            BaseInteraction.check_same_shape(
                first,
                second,
            )

            BaseInteraction.check_same_device(
                first,
                second,
            )

            BaseInteraction.check_same_dtype(
                first,
                second,
            )

            BaseInteraction.ensure_finite(
                first.tensor,
            )

            BaseInteraction.ensure_finite(
                second.tensor,
            )

            weights = torch.softmax(
                torch.stack(
                (
                    self.weight_a,
                    self.weight_b,
                )
            ),
            dim=0,
        )

            fused = (
                weights[0] * first.tensor
                + weights[1] * second.tensor
            )

            residual = fused

            fused = self.projection(fused)

            fused = (
                fused
                * self.layer_scale.view(
                    1,
                    -1,
                    1,
                    1,
                )
                + residual
            )

            BaseInteraction.ensure_finite(fused)

            return BaseInteraction.clone_feature(
                first,
                fused,
            )

    def forward(
        self,
        first: StageRepresentation,
        second: StageRepresentation,
    ) -> StageRepresentation:
        """
        Fuse two stage representations.
        """

        return StageRepresentation(
            stage=first.stage,
            spatial=self._fuse_feature(
                first.spatial,
                second.spatial,
            ),
            frequency=self._fuse_feature(
                first.frequency,
                second.frequency,
            ),
            context=self._fuse_feature(
                first.context,
                second.context,
            ),
            fused=self._fuse_feature(
                first.fused,
                second.fused,
            ),
        )
import torch.nn as nn

class InteractionFusion(nn.Module):
   

    def __init__(
        self,
        stage_channels: tuple[int, ...] = (
            96,
            192,
            384,
            768,
        ),
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        if len(stage_channels) != 4:
            raise InteractionFusionError(
                "Exactly four stage channel sizes are required."
            )

        self.blocks = nn.ModuleDict(
            {
                f"stage{i + 1}": InteractionFusionBlock(
                    channels=channels,
                    dropout=dropout,
                )
                for i, channels in enumerate(stage_channels)
            }
        )

    def forward(
        self,
        attention_output: RepresentationOutput,
        gated_output: RepresentationOutput,
    ) -> RepresentationOutput:
        """
        Fuse two RepresentationOutput objects.
        """

        attention_output.validate()
        gated_output.validate()

        fused_stages: list[StageRepresentation] = []

        for attention_stage, gated_stage in zip(
            attention_output.stages,
            gated_output.stages,
        ):
            if attention_stage.stage != gated_stage.stage:
                raise InteractionFusionError(
                    "Stage ordering mismatch between "
                    "interaction outputs."
                )

            InteractionValidator.validate_shape_consistency(
                attention_stage,
            )
            InteractionValidator.validate_device_consistency(
                attention_stage,
            )
            InteractionValidator.validate_dtype_consistency(
                attention_stage,
            )
            InteractionValidator.validate_channel_consistency(
                attention_stage,
            )
            InteractionValidator.validate_batch_consistency(
                attention_stage,
            )

            InteractionValidator.validate_shape_consistency(
                gated_stage,
            )
            InteractionValidator.validate_device_consistency(
                gated_stage,
            )
            InteractionValidator.validate_dtype_consistency(
                gated_stage,
            )
            InteractionValidator.validate_channel_consistency(
                gated_stage,
            )
            InteractionValidator.validate_batch_consistency(
                gated_stage,
            )

            if attention_stage.stage not in self.blocks:
                raise InteractionFusionError(
                    f"Unknown stage '{attention_stage.stage}'."
                )

            block = self.blocks[
                attention_stage.stage
            ]

            stage = block(
                attention_stage,
                gated_stage,
            )

            stage.validate()

            fused_stages.append(stage)

        output = RepresentationOutput(
            stage1=fused_stages[0],
            stage2=fused_stages[1],
            stage3=fused_stages[2],
            stage4=fused_stages[3],
            original_size=attention_output.original_size,
        )

        output.validate()

        return output
    
    def extra_repr(self) -> str:
        
        return (
            f"stages={list(self.blocks.keys())}, "
            f"num_blocks={len(self.blocks)}"
        )
