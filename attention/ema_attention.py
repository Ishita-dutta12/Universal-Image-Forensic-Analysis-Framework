
"""
Efficient Multi-scale Attention (EMA)

Applies Efficient Multi-scale Attention to backbone feature maps.

Reference:
Efficient Multi-Scale Attention Module with Cross-Spatial Learning
(ICASSP 2023)

Author: Ishita Dutta
Project: UIFAF
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .exceptions import AttentionError


class EMAAttention(nn.Module):
    """
    Efficient Multi-scale Attention.

    Parameters
    ----------
    channels : int
        Number of feature channels.

    groups : int, default=8
        Number of channel groups.
    """

    def __init__(
        self,
        channels: int,
        groups: int = 8,
    ):
        super().__init__()

        if channels <= 0:
            raise ValueError(
                "channels must be positive."
            )

        if groups <= 0:
            raise ValueError(
                "groups must be positive."
            )

        if channels % groups != 0:
            raise ValueError(
                "channels must be divisible by groups."
            )

        self.channels = channels
        self.groups = groups
        self.group_channels = channels // groups

        self.pool_h = nn.AdaptiveAvgPool2d(
            (None, 1)
        )

        self.pool_w = nn.AdaptiveAvgPool2d(
            (1, None)
        )

        self.conv1x1 = nn.Conv2d(
            self.group_channels,
            self.group_channels,
            kernel_size=1,
            bias=False,
        )

        self.conv3x3 = nn.Conv2d(
            self.group_channels,
            self.group_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.group_norm = nn.GroupNorm(
            1,
            self.group_channels,
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor
            Shape (B,C,H,W)

        Returns
        -------
        enhanced_features
        attention_map
        """

        if x.ndim != 4:
            raise AttentionError(
                "Expected input shape (B,C,H,W)."
            )

        b, c, h, w = x.shape
        if h == 0 or w == 0:
            raise AttentionError(
            "Input spatial dimensions must be positive."
        )
        
        if c != self.channels:
            raise AttentionError(
                "Input channel mismatch."
            )

        # -------------------------------------------------
        # Split into groups
        # -------------------------------------------------

        x = x.contiguous().view(
            b * self.groups,
            self.group_channels,
            h,
            w,
        )

        # -------------------------------------------------
        # Multi-scale pooling
        # -------------------------------------------------

        x_h = self.pool_h(x)

        x_w = self.pool_w(x)

        x_w = x_w.transpose(
            2,
            3,
        )

        hw = torch.cat(
            [
                x_h,
                x_w,
            ],
            dim=2,
        )

        hw = self.conv1x1(hw)

        x_h, x_w = torch.split(
            hw,
            [h, w],
            dim=2,
        )

        x_w = x_w.transpose(
            2,
            3,
        )

        # -------------------------------------------------
        # Spatial interaction
        # -------------------------------------------------

        spatial = (
            x
            * torch.sigmoid(x_h)
            * torch.sigmoid(x_w)
        )

        spatial = self.group_norm(
            spatial
        )

        # -------------------------------------------------
        # Local interaction
        # -------------------------------------------------

        local = self.conv3x3(x)

        # -------------------------------------------------
        # Global descriptors
        # -------------------------------------------------

        spatial_weight = F.adaptive_avg_pool2d(
            spatial,
            1,
        ).reshape(
            b * self.groups,
            1,
            self.group_channels,
        )

        local_weight = F.adaptive_avg_pool2d(
            local,
            1,
        ).reshape(
            b * self.groups,
            1,
            self.group_channels,
        )

        spatial_weight = self.softmax(
            spatial_weight
        )

        local_weight = self.softmax(
            local_weight
        )

        spatial_flat = spatial.contiguous().view(
            b * self.groups,
            self.group_channels,
            -1,
        )

        local_flat = local.contiguous().view(
            b * self.groups,
            self.group_channels,
            -1,
        )

        attention = (
            torch.matmul(
                spatial_weight,
                local_flat,
            )
            +
            torch.matmul(
                local_weight,
                spatial_flat,
            )
        )

        attention = attention.contiguous().view(
            b * self.groups,
            1,
            h,
            w,
        )

        attention = torch.sigmoid(
            attention
        )

        enhanced = (
            x
            * attention
        )

        enhanced = enhanced.reshape(
            b,
            c,
            h,
            w,
        )

        attention_map = attention.reshape(
            b,
            self.groups,
            h,
            w,
        )
        if not torch.isfinite(enhanced).all():
            raise AttentionError(
                "Enhanced features contain NaN or Inf."
            )

        if not torch.isfinite(attention_map).all():
            raise AttentionError(
                "Attention map contains NaN or Inf."
            )
        return (
            enhanced,
            attention_map,
        )

    def extra_repr(
        self,
    ) -> str:

        return (
            f"channels={self.channels}, "
            f"groups={self.groups}"
        )