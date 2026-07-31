"""
Discrete Wavelet Transform (PyTorch).

Implements one-level 2D Haar DWT using grouped convolutions.
Fully differentiable and GPU compatible.

Author: Ishita Dutta
Project: UIFAF
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
import torch.nn.functional as F

from .exceptions import RepresentationError


@dataclass(frozen=True)
class DWTCoefficients:
    """
    Stores one-level DWT coefficients.
    """

    ll: torch.Tensor
    lh: torch.Tensor
    hl: torch.Tensor
    hh: torch.Tensor

    def validate(self) -> None:

        for name, tensor in (
            ("LL", self.ll),
            ("LH", self.lh),
            ("HL", self.hl),
            ("HH", self.hh),
        ):

            if not isinstance(tensor, torch.Tensor):
                raise RepresentationError(
                    f"{name} must be a torch.Tensor."
                )

            if tensor.ndim != 4:
                raise RepresentationError(
                    f"{name} must be 4-dimensional."
                )

            if not torch.isfinite(tensor).all():
                raise RepresentationError(
                    f"{name} contains NaN or Inf."
                )


class DWTTransform:
    """
    Differentiable one-level Haar DWT.
    """

    def __init__(
        self,
        wavelet: str = "haar",
    ):

        if wavelet.lower() != "haar":
            raise RepresentationError(
                "Currently only Haar wavelet is supported."
            )

        s = 1.0 / sqrt(2.0)

        low = torch.tensor(
            [s, s],
            dtype=torch.float32,
        )

        high = torch.tensor(
            [s, -s],
            dtype=torch.float32,
        )

        self.ll_kernel = torch.outer(
            low,
            low,
        )

        self.lh_kernel = torch.outer(
            low,
            high,
        )

        self.hl_kernel = torch.outer(
            high,
            low,
        )

        self.hh_kernel = torch.outer(
            high,
            high,
        )

    def _apply_filter(
        self,
        x: torch.Tensor,
        kernel: torch.Tensor,
    ) -> torch.Tensor:

        channels = x.shape[1]

        kernel = (
            kernel.view(1, 1, 2, 2)
            .repeat(channels, 1, 1, 1)
            .to(
                device=x.device,
                dtype=x.dtype,
            )
        )

        return F.conv2d(
            x,
            kernel,
            stride=2,
            groups=channels,
        )

    def __call__(
        self,
        x: torch.Tensor,
    ) -> DWTCoefficients:

        if not isinstance(
            x,
            torch.Tensor,
        ):
            raise RepresentationError(
                "Input must be a torch.Tensor."
            )

        if x.ndim != 4:
            raise RepresentationError(
                "Input tensor must be 4-dimensional."
            )

        coeffs = DWTCoefficients(

            ll=self._apply_filter(
                x,
                self.ll_kernel,
            ),

            lh=self._apply_filter(
                x,
                self.lh_kernel,
            ),

            hl=self._apply_filter(
                x,
                self.hl_kernel,
            ),

            hh=self._apply_filter(
                x,
                self.hh_kernel,
            ),
        )

        coeffs.validate()

        return coeffs