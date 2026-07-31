"""
Cross Representation Multi-Attention (MRCA).

This module enables information exchange between multiple
representation domains (Spatial, Frequency, Context and Fused)
within each backbone stage.

The module performs lightweight convolutional cross-attention
while preserving spatial feature maps.

Pipeline
--------
RepresentationOutput
        │
        ▼
CrossAttention
        │
        ▼
Stage-wise MRCA
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
import torch.nn.functional as F

from representation.representation_models import (
    RepresentationFeature,
    StageRepresentation,
    RepresentationOutput,

)

from .base_interaction import BaseInteraction
from .exceptions import (
    CrossAttentionError,
)
from .interaction_validator import InteractionValidator


class CrossAttentionBlock(nn.Module):
    """
    Multi-Representation Cross Attention block.

    The block exchanges information between all available
    representations belonging to the same backbone stage.

    Notes
    -----
    • Operates directly on feature maps.
    • Preserves spatial resolution.
    • Uses lightweight 1×1 convolutions.
    • Residual learning with LayerScale.
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

        # --------------------------------------------------
        # Query Projection
        # --------------------------------------------------

        self.query_projection = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )

        # --------------------------------------------------
        # Key Projection
        # --------------------------------------------------

        self.key_projection = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )

        # --------------------------------------------------
        # Value Projection
        # --------------------------------------------------

        self.value_projection = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )

        # --------------------------------------------------
        # Output Projection
        # --------------------------------------------------

        self.output_projection = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )

        # --------------------------------------------------
        # LayerScale
        # --------------------------------------------------

        self.layer_scale = nn.Parameter(
            torch.ones(channels) * 1e-4
        )

        # --------------------------------------------------
        # Learnable Representation Importance
        # --------------------------------------------------

        self.spatial_weight = nn.Parameter(
            torch.tensor(1.0)
        )

        self.frequency_weight = nn.Parameter(
            torch.tensor(1.0)
        )

        self.context_weight = nn.Parameter(
            torch.tensor(1.0)
        )

        self.fused_weight = nn.Parameter(
            torch.tensor(1.0)
        )
    def _compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
       

        batch_size, channels, height, width = query.shape

        # ---------------------------------------------
        # Flatten spatial dimensions
        #
        # Query : (B, HW, C)
        # Key   : (B, C, HW)
        # Value : (B, HW, C)
        # ---------------------------------------------

        query = query.flatten(2).transpose(1, 2)

        key = key.flatten(2)

        value = value.flatten(2).transpose(1, 2)

        # ---------------------------------------------
        # Scaled Dot-Product Attention
        # ---------------------------------------------

        attention = torch.bmm(
            query,
            key,
        )

        attention = attention / (channels ** 0.5)

        attention = F.softmax(
            attention,
            dim=-1,
        )

        # ---------------------------------------------
        # Aggregate Values
        # ---------------------------------------------

        output = torch.bmm(
            attention,
            value,
        )

        output = (
            output.transpose(1, 2)
            .reshape(
                batch_size,
                channels,
                height,
                width,
            )
        )

        return output
    
    def forward(
        self,
        spatial: RepresentationFeature,
        frequency: Optional[RepresentationFeature],
        context: Optional[RepresentationFeature],
        fused: Optional[RepresentationFeature],
    ) -> tuple[
        RepresentationFeature,
        Optional[RepresentationFeature],
        Optional[RepresentationFeature],
        Optional[RepresentationFeature],
    ]:
        

        try:

            features: list[
                tuple[
                str,
                RepresentationFeature,
                nn.Parameter,
                ]
            ] = [
                (
                    "spatial",
                    spatial,
                    self.spatial_weight,
                )
            ]

            if frequency is not None:
                features.append(
                    (
                        "frequency",
                        frequency,
                        self.frequency_weight,
                    )
                )

            if context is not None:
                features.append(
                    (
                        "context",
                        context,
                        self.context_weight,
                    )
                )

            if fused is not None:
                features.append(
                    (
                        "fused",
                        fused,
                        self.fused_weight,
                    )
                )

            # -----------------------------------------
            # Normalize representation importance
            # -----------------------------------------

            raw_weights = torch.stack(
                [weight for _, _, weight in features]
            )

            normalized_weights = F.softmax(
                raw_weights,
                dim=0,
            )

            outputs: dict[
                str,
                RepresentationFeature,
            ] = {}

            # -----------------------------------------
            # Update every representation
            # -----------------------------------------

            for name, feature, _ in features:

                current = feature.tensor

                query = self.query_projection(
                    current
                )

                attention_sum = torch.zeros_like(
                    query
                )

                # -------------------------------------
                # Attend to every available feature
                # -------------------------------------

                for index, (_, other, _) in enumerate(
                    features
                ):

                    key = self.key_projection(
                        other.tensor
                    )

                    value = self.value_projection(
                        other.tensor
                    )

                    attended = self._compute_attention(
                        query=query,
                        key=key,
                        value=value,
                    )

                    attention_sum = (
                        attention_sum
                        + normalized_weights[index]
                        * attended
                    )

                attention_sum = (
                    self.output_projection(
                        attention_sum
                    )
                )

                refined = (
                    current
                    + self.layer_scale.view(
                        1,
                        -1,
                        1,
                        1,
                    )
                    * attention_sum
                )

                BaseInteraction.ensure_finite(
                    refined
                )

                outputs[name] = (
                    BaseInteraction.clone_feature(
                        feature,
                        refined,
                    )
                )

            return (
                outputs["spatial"],
                outputs.get("frequency"),
                outputs.get("context"),
                outputs.get("fused"),
            )

        except CrossAttentionError:
            raise

        except Exception as exc:
            raise CrossAttentionError(
                str(exc)
            ) from exc
            
class CrossAttention(BaseInteraction):
    

    def __init__(
        self,
        stage_channels: tuple[int, int, int, int] = (
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
            raise CrossAttentionError(
                "Exactly four stage channel dimensions are required."
            )

        self.blocks = nn.ModuleDict(
            {
                stage: CrossAttentionBlock(
                    channels=channels,
                    reduction=reduction,
                    dropout=dropout,
                )
                for stage, channels in zip(
                    self.STAGES,
                    stage_channels,
                )
            }
        )

    def interact_stage(
        self,
        stage: StageRepresentation,
    ) -> StageRepresentation:
        """
        Apply cross attention to one backbone stage.
        """

        InteractionValidator.validate_shape_consistency(stage)
        InteractionValidator.validate_device_consistency(stage)
        InteractionValidator.validate_dtype_consistency(stage)
        InteractionValidator.validate_channel_consistency(stage)
        InteractionValidator.validate_batch_consistency(stage)

        if stage.stage not in self.blocks:
            raise CrossAttentionError(
            f"Unknown stage '{stage.stage}'."
        )

        block = self.blocks[stage.stage]

        (
            spatial,
            frequency,
            context,
            fused,
        ) = block(
            spatial=stage.spatial,
            frequency=stage.frequency,
            context=stage.context,
            fused=stage.fused,
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
        Extra representation shown by PyTorch.
        """

        return (
            f"stages={list(self.blocks.keys())}, "
            f"num_blocks={len(self.blocks)}"
        )