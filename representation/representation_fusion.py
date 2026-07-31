"""
Representation Fusion Module.

Fuses spatial, frequency and context representations
into unified forensic representations for each backbone stage.

Author: Ishita Dutta
Project: UIFAF
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .representation_models import (
    RepresentationFeature,
    StageRepresentation,
)

from .exceptions import (
    InvalidRepresentationFeatureError,
)
class ChannelAttention(nn.Module):
    """
    Lightweight channel attention used after
    feature fusion.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
    ) -> None:

        super().__init__()

        hidden = max(channels // reduction, 1)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 4:
            raise InvalidRepresentationFeatureError(
                f"Expected a 4D tensor, got shape {tuple(x.shape)}."
            )

        b, c, _, _ = x.shape

        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)

        return x * w
    
class ResidualFusionBlock(nn.Module):
    

    def __init__(
        self,
        channels: int,
    ) -> None:

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(channels),

            nn.GELU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(channels),
        )

        self.act = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        residual = x

        x = self.block(x)

        return self.act(x + residual)
class RepresentationFusion(nn.Module):
    """
    Fuses spatial, frequency and context
    representations into a unified
    StageRepresentation.
    """

    REQUIRED_STAGES = (
        "stage1",
        "stage2",
        "stage3",
        "stage4",
    )

    def __init__(
        self,
        stage_channels: Dict[str, int],
        reduction: int = 16,
    ) -> None:

        super().__init__()

        missing = [
            stage
            for stage in self.REQUIRED_STAGES
            if stage not in stage_channels
        ]

        if missing:
            raise ValueError(
                f"Missing stage definitions: {missing}"
            )

        self.projection = nn.ModuleDict({
            stage: nn.Conv2d(
                stage_channels[stage] * 3,
                stage_channels[stage],
                kernel_size=1,
                bias=False,
            )
            for stage in self.REQUIRED_STAGES
        })

        self.attention = nn.ModuleDict({
            stage: ChannelAttention(
                stage_channels[stage],
                reduction=reduction,
            )
            for stage in self.REQUIRED_STAGES
        })

        self.refinement = nn.ModuleDict({
            stage: ResidualFusionBlock(
                stage_channels[stage],
            )
            for stage in self.REQUIRED_STAGES
        })

        self._initialize_weights()

    def _initialize_weights(self) -> None:

        for module in self.modules():

            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(
                    module.weight,
                    std=0.02,
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _fuse_stage(
        self,
        spatial: RepresentationFeature,
        frequency: RepresentationFeature,
        context: RepresentationFeature,
    ) -> StageRepresentation:

        spatial.validate()
        frequency.validate()
        context.validate()
        if (
            spatial.tensor.shape != frequency.tensor.shape
            or spatial.tensor.shape != context.tensor.shape
    ):
            raise InvalidRepresentationFeatureError(
            "Representation tensors must have identical shapes."
        )
        stage = spatial.stage

        if (
            frequency.stage != stage
            or context.stage != stage
        ):
            raise InvalidRepresentationFeatureError(
                f"Stage mismatch while fusing {stage}."
            )

        x = torch.cat(
            (
                spatial.tensor,
                frequency.tensor,
                context.tensor,
            ),
            dim=1,
        )

        x = self.projection[stage](x)
        x = self.attention[stage](x)
        x = self.refinement[stage](x)

        fused = RepresentationFeature(
            stage=stage,
            tensor=x,
            representation_type="fused",
        )

        fused.validate()

        return StageRepresentation(
            stage=stage,
            spatial=spatial,
            frequency=frequency,
            context=context,
            fused=fused,
        )
    def forward(
        self,
        spatial: Dict[str, RepresentationFeature],
        frequency: Dict[str, RepresentationFeature],
        context: Dict[str, RepresentationFeature],
        ) -> Dict[str, StageRepresentation]:
    

        return {
        stage: self._fuse_stage(
            spatial[stage],
            frequency[stage],
            context[stage],
        )
        for stage in self.REQUIRED_STAGES
    }