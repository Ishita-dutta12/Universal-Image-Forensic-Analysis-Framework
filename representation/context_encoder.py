
"""
Context Representation Encoder.

Extracts high-level semantic and contextual representations
from attention feature maps.

Author: Ishita Dutta
Project: UIFAF
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from attention.attention_models import AttentionFeatureMap
import representation
#from tests.test_context_encoder import attention_output

from .base_encoder import BaseRepresentationEncoder
#from .representation_models import RepresentationFeature
from .exceptions import (
    InvalidRepresentationFeatureError,
    RepresentationInitializationError,
)
#from attention.attention_models import AttentionOutput
from attention.attention_models import (
    AttentionFeatureMap,
    AttentionOutput,
)
from .representation_models import (
    RepresentationOutput,
    StageRepresentation,
)
from .representation_models import (
    RepresentationFeature,
    RepresentationOutput,
    StageRepresentation,
)

class GlobalContextGate(nn.Module):
    """
    Lightweight channel attention module that emphasizes
    globally important semantic features.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
    ) -> None:

        super().__init__()

        reduced = max(channels // reduction, 1)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, reduced),
            nn.GELU(),
            nn.Linear(reduced, channels),
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
class ContextBlock(nn.Module):
    """
    ConvNeXt-inspired context extraction block.
    """

    def __init__(
        self,
        channels: int,
        expansion: int = 4,
    ) -> None:

        super().__init__()

        hidden = channels * expansion

        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
            bias=False,
        )

        self.norm = nn.LayerNorm(channels)

        self.pw1 = nn.Linear(
            channels,
            hidden,
        )

        self.act = nn.GELU()

        self.pw2 = nn.Linear(
            hidden,
            channels,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # Validate input
        if x.ndim != 4:
            raise InvalidRepresentationFeatureError(
                f"Expected a 4D tensor, got shape {tuple(x.shape)}."
            )

        residual = x

        x = self.dwconv(x)

        x = x.permute(0, 2, 3, 1)

        x = self.norm(x)

        x = self.pw1(x)

        x = self.act(x)

        x = self.pw2(x)

        x = x.permute(0, 3, 1, 2)

        return x + residual


class ContextEncoder(BaseRepresentationEncoder):
    """
    Context representation encoder.

    Learns high-level semantic representations from
    attention feature maps.
    """

    def __init__(
        self,
        stage_channels: Dict[str, int],
        expansion: int = 4,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        required = (
            "stage1",
            "stage2",
            "stage3",
            "stage4",
        )

        missing = [
            stage
        for stage in required
            if stage not in stage_channels
        ]

        if missing:
            raise RepresentationInitializationError(
        f"Missing stage definitions: {missing}"
    )
        self.blocks = nn.ModuleDict({
            stage: ContextBlock(
                channels=channels,
                expansion=expansion,
            )
            for stage, channels in stage_channels.items()
        })

        self.gates = nn.ModuleDict({
            stage: GlobalContextGate(
                channels=channels,
                reduction=reduction,
            )
            for stage, channels in stage_channels.items()
        })

        

        self._initialize_weights()

    def _encode_stage(
        self,
        feature: AttentionFeatureMap,
    ) -> RepresentationFeature:

        feature.validate()

        stage = feature.stage

        x = self.blocks[stage](feature.tensor)
        x = self.gates[stage](x)

        if not torch.isfinite(x).all():
            raise InvalidRepresentationFeatureError(
                f"{stage}: Context representation contains NaN or Inf."
        )

        representation = RepresentationFeature(
            stage=stage,
            tensor=x,
            representation_type="context",
        )

        representation.validate()

        return representation

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(
                    module.weight,
                    std=0.02,
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                
        def forward(
            self,
            attention_output: AttentionOutput,
            ) -> RepresentationOutput:

            return RepresentationOutput(

            stage1=StageRepresentation(
            stage="stage1",
            context=self._encode_stage(
                attention_output.stage1,
            ),
        ),

        stage2=StageRepresentation(
            stage="stage2",
            context=self._encode_stage(
                attention_output.stage2,
            ),
        ),

        stage3=StageRepresentation(
            stage="stage3",
            context=self._encode_stage(
                attention_output.stage3,
            ),
        ),

        stage4=StageRepresentation(
            stage="stage4",
            context=self._encode_stage(
                attention_output.stage4,
            ),
        ),

        original_size=attention_output.original_size,
    )