
"""
Tests for Efficient Multi-scale Attention (EMA).

Author: Ishita Dutta
Project: UIFAF
"""

import pytest
import torch

from attention.ema_attention import EMAAttention
from attention.exceptions import AttentionError


def test_initialization():
    """
    EMA should initialize correctly.
    """

    ema = EMAAttention(96)

    assert ema.channels == 96
    assert ema.groups == 8
    assert ema.group_channels == 12


def test_invalid_channels():
    """
    Channels must be positive.
    """

    with pytest.raises(ValueError):
        EMAAttention(0)


def test_invalid_groups():
    """
    Groups must be positive.
    """

    with pytest.raises(ValueError):
        EMAAttention(
            channels=96,
            groups=0,
        )


def test_invalid_divisibility():
    """
    Channels must be divisible by groups.
    """

    with pytest.raises(ValueError):
        EMAAttention(
            channels=95,
            groups=8,
        )


def test_forward_shape():
    """
    Output feature map should preserve shape.
    """

    ema = EMAAttention(96)

    x = torch.randn(
        2,
        96,
        56,
        56,
    )

    enhanced, attention = ema(x)

    assert enhanced.shape == x.shape

    assert attention.shape == (
        2,
        8,
        56,
        56,
    )


def test_output_is_finite():
    """
    Output should not contain NaN or Inf.
    """

    ema = EMAAttention(96)

    x = torch.randn(
        2,
        96,
        56,
        56,
    )

    enhanced, attention = ema(x)

    assert torch.isfinite(enhanced).all()

    assert torch.isfinite(attention).all()


def test_backward():
    """
    Gradients should propagate.
    """

    ema = EMAAttention(96)

    x = torch.randn(
        2,
        96,
        56,
        56,
        requires_grad=True,
    )

    enhanced, _ = ema(x)

    loss = enhanced.mean()

    loss.backward()

    assert x.grad is not None


def test_invalid_dimensions():
    """
    Input must be BCHW.
    """

    ema = EMAAttention(96)

    x = torch.randn(
        96,
        56,
        56,
    )

    with pytest.raises(AttentionError):
        ema(x)


def test_channel_mismatch():
    """
    Channel mismatch should raise.
    """

    ema = EMAAttention(96)

    x = torch.randn(
        1,
        64,
        56,
        56,
    )

    with pytest.raises(AttentionError):
        ema(x)


def test_zero_input():
    """
    Zero input should remain numerically stable.
    """

    ema = EMAAttention(96)

    x = torch.zeros(
        1,
        96,
        56,
        56,
    )

    enhanced, attention = ema(x)

    assert torch.isfinite(enhanced).all()

    assert torch.isfinite(attention).all()


def test_extra_repr():
    """
    Module representation should include configuration.
    """

    ema = EMAAttention(96)

    text = ema.extra_repr()

    assert "channels=96" in text
    assert "groups=8" in text