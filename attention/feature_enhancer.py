"""
Feature enhancer for the Attention module.

Applies EMA Attention independently to each backbone stage.

Author: Ishita Dutta
Project: UIFAF
"""

from __future__ import annotations

import torch.nn as nn

from backbone.backbone_models import BackboneOutput

from .attention_models import (
    AttentionFeatureMap,
    AttentionOutput,
)
from .ema_attention import EMAAttention


class FeatureEnhancer(nn.Module):
    """
    Applies EMA Attention to all backbone feature maps.
    """

    def __init__(self, groups: int = 8):
        super().__init__()

        # ConvNeXt V2 Tiny output channels
        self.stage1_attention = EMAAttention(96, groups)
        self.stage2_attention = EMAAttention(192, groups)
        self.stage3_attention = EMAAttention(384, groups)
        self.stage4_attention = EMAAttention(768, groups)

    def forward(
        self,
        backbone_output: BackboneOutput,
    ) -> AttentionOutput:
        """
        Enhance all backbone feature maps using EMA.
        """

        backbone_output.validate()

        # -----------------------------
        # Stage 1
        # -----------------------------

        enhanced1, attn1 = self.stage1_attention(
            backbone_output.stage1.tensor
        )

        stage1 = AttentionFeatureMap(
            stage=backbone_output.stage1.stage,
            tensor=enhanced1,
            channels=backbone_output.stage1.channels,
            stride=backbone_output.stage1.stride,
            attention_map=attn1,
            attention_name="EMA",
        )

        # -----------------------------
        # Stage 2
        # -----------------------------

        enhanced2, attn2 = self.stage2_attention(
            backbone_output.stage2.tensor
        )

        stage2 = AttentionFeatureMap(
            stage=backbone_output.stage2.stage,
            tensor=enhanced2,
            channels=backbone_output.stage2.channels,
            stride=backbone_output.stage2.stride,
            attention_map=attn2,
            attention_name="EMA",
        )

        # -----------------------------
        # Stage 3
        # -----------------------------

        enhanced3, attn3 = self.stage3_attention(
            backbone_output.stage3.tensor
        )

        stage3 = AttentionFeatureMap(
            stage=backbone_output.stage3.stage,
            tensor=enhanced3,
            channels=backbone_output.stage3.channels,
            stride=backbone_output.stage3.stride,
            attention_map=attn3,
            attention_name="EMA",
        )

        # -----------------------------
        # Stage 4
        # -----------------------------

        enhanced4, attn4 = self.stage4_attention(
            backbone_output.stage4.tensor
        )

        stage4 = AttentionFeatureMap(
            stage=backbone_output.stage4.stage,
            tensor=enhanced4,
            channels=backbone_output.stage4.channels,
            stride=backbone_output.stage4.stride,
            attention_map=attn4,
            attention_name="EMA",
        )

        output = AttentionOutput(
            stage1=stage1,
            stage2=stage2,
            stage3=stage3,
            stage4=stage4,
            original_size=backbone_output.original_size,
        )

        output.validate()

        return output