"""
Tests for ConvNeXt V2 Backbone.

Author: Ishita Dutta
Project: UIFAF
"""

import pytest
import torch
import torch.nn as nn

from backbone.convnext_v2 import ConvNeXtV2Backbone
from backbone.exceptions import (
    InvalidBackboneOutputError,
)


def test_backbone_initialization():
    """
    Backbone should initialize successfully.
    """
    backbone = ConvNeXtV2Backbone(pretrained=False)

    assert isinstance(backbone, nn.Module)


def test_forward_output():
    """
    Backbone should return four feature maps.
    """
    backbone = ConvNeXtV2Backbone(pretrained=False)

    x = torch.randn(1, 3, 224, 224)

    outputs = backbone(x)

    assert isinstance(outputs, list)
    assert len(outputs) == 4

    for feature in outputs:
        assert isinstance(feature, torch.Tensor)
        assert feature.ndim == 4


def test_feature_map_sizes():
    """
    Spatial resolution should decrease stage-by-stage.
    """
    backbone = ConvNeXtV2Backbone(pretrained=False)

    x = torch.randn(1, 3, 224, 224)

    outputs = backbone(x)

    heights = [f.shape[2] for f in outputs]
    widths = [f.shape[3] for f in outputs]

    assert heights == sorted(heights, reverse=True)
    assert widths == sorted(widths, reverse=True)


def test_channels_property():
    """
    Backbone should expose feature channels.
    """
    backbone = ConvNeXtV2Backbone(pretrained=False)

    channels = backbone.channels

    assert isinstance(channels, list)
    assert len(channels) == 4

    for c in channels:
        assert isinstance(c, int)
        assert c > 0


def test_reductions_property():
    """
    Backbone should expose feature reductions.
    """
    backbone = ConvNeXtV2Backbone(pretrained=False)

    reductions = backbone.reductions

    assert isinstance(reductions, list)
    assert len(reductions) == 4

    for r in reductions:
        assert isinstance(r, int)
        assert r > 0


def test_freeze():
    """
    Freeze should disable gradients.
    """
    backbone = ConvNeXtV2Backbone(pretrained=False)

    backbone.freeze()

    assert all(
        not p.requires_grad
        for p in backbone.parameters()
    )


def test_unfreeze():
    """
    Unfreeze should enable gradients.
    """
    backbone = ConvNeXtV2Backbone(pretrained=False)

    backbone.freeze()
    backbone.unfreeze()

    assert all(
        p.requires_grad
        for p in backbone.parameters()
    )


def test_invalid_number_of_outputs():
    """
    Should raise if model returns an invalid number of feature maps.
    """

    backbone = ConvNeXtV2Backbone(pretrained=False)

    class FakeModel(nn.Module):
        def forward(self, x):
            return [torch.randn(1, 3, 8, 8)]

    backbone.model = FakeModel()

    with pytest.raises(InvalidBackboneOutputError):
        backbone(torch.randn(1, 3, 224, 224))