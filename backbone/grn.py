"""
Global Response Normalization (GRN)

Reference:
ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders
https://arxiv.org/abs/2301.00808

This module implements the GRN layer exactly as described in the paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GRN(nn.Module):
    """
    Global Response Normalization.

    Input Shape
    -----------
    (B, C, H, W)

    Output Shape
    ------------
    (B, C, H, W)

    Formula
    -------
    Gx = ||x||₂

    Nx = Gx / mean(Gx)

    y = γ * (x * Nx) + β + x
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):

    #Parameters
    
    #num_channels : int
        #Number of feature channels.

    #eps : float
        #Small constant for numerical stability.
    
        super().__init__()

        if num_channels <= 0:
            raise ValueError(
            "num_channels must be positive."
        )

        self.eps = eps

        self.gamma = nn.Parameter(
            torch.zeros(1, num_channels, 1, 1)
        )

        self.beta = nn.Parameter(
            torch.zeros(1, num_channels, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            Shape (B, C, H, W)

        Returns
        -------
        Tensor
            Shape (B, C, H, W)
        """

        # Global L2 norm across spatial dimensions
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)

        # Normalize response
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)

        # Apply GRN
        y = self.gamma * (x * nx)

        y = y + self.beta + x

        return y

    def extra_repr(self) -> str:
        return (
            f"channels={self.gamma.shape[1]}, "
            f"eps={self.eps}"
        )