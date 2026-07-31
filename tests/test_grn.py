"""
Tests for Global Response Normalization (GRN)

Author: Ishita Dutta
Project: UIFAF
"""

import pytest
import torch

from backbone.grn import GRN


def test_grn_initialization():
    """
    GRN should initialize learnable parameters correctly.
    """
    channels = 96
    grn = GRN(channels)

    assert grn.gamma.shape == (1, channels, 1, 1)
    assert grn.beta.shape == (1, channels, 1, 1)


def test_grn_forward_shape():
    """
    GRN should preserve input shape.
    """
    grn = GRN(96)

    x = torch.randn(2, 96, 56, 56)

    output = grn(x)

    assert output.shape == x.shape


def test_grn_output_is_finite():
    """
    Output should not contain NaN or Inf.
    """
    grn = GRN(96)

    x = torch.randn(4, 96, 56, 56)

    output = grn(x)

    assert torch.isfinite(output).all()


def test_grn_backward():
    """
    Gradients should flow correctly.
    """
    grn = GRN(96)

    x = torch.randn(
        2,
        96,
        56,
        56,
        requires_grad=True,
    )

    output = grn(x)

    loss = output.mean()

    loss.backward()

    assert x.grad is not None


def test_grn_zero_input():
    """
    Zero tensor should produce finite output.
    """
    grn = GRN(96)

    x = torch.zeros(1, 96, 56, 56)

    output = grn(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_invalid_channel_size():
    """
    Creating GRN with invalid channels should fail.
    """
    with pytest.raises(ValueError):
        GRN(0)