"""
Tests for DWT Transform.

Author: Ishita Dutta
Project: UIFAF
"""

import torch

from representation.dwt_transform import (
    DWTTransform,
)


def test_dwt_shapes():

    x = torch.randn(
        2,
        96,
        32,
        32,
    )

    dwt = DWTTransform()

    coeffs = dwt(x)

    assert coeffs.ll.shape == (
        2,
        96,
        16,
        16,
    )

    assert coeffs.lh.shape == (
        2,
        96,
        16,
        16,
    )

    assert coeffs.hl.shape == (
        2,
        96,
        16,
        16,
    )

    assert coeffs.hh.shape == (
        2,
        96,
        16,
        16,
    )