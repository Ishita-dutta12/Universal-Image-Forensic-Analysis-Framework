"""
Tests for Backbone Validator

Author: Ishita Dutta
Project: UIFAF
"""

import pytest
import torch

from backbone.backbone_models import FeatureMap, BackboneOutput
from backbone.backbone_validator import BackboneValidator
from backbone.exceptions import (
    InvalidBackboneOutputError,
    InvalidFeatureMapError,
)


def create_feature_map(
    channels: int,
    height: int,
    width: int,
    stride: int,
) -> FeatureMap:
    tensor = torch.randn(1, channels, height, width)

    return FeatureMap(
        stage=f"stage{stride}",
        tensor=tensor,
        channels=channels,
        stride=stride,
    )


def create_backbone_output() -> BackboneOutput:
    return BackboneOutput(
        stage1=create_feature_map(96, 56, 56, 4),
        stage2=create_feature_map(192, 28, 28, 8),
        stage3=create_feature_map(384, 14, 14, 16),
        stage4=create_feature_map(768, 7, 7, 32),
        original_size=(224, 224),
    )


def test_valid_backbone_output():
    output = create_backbone_output()

    BackboneValidator.validate(output)


def test_invalid_original_size():
    output = create_backbone_output()

    output.original_size = (0, 224)

    with pytest.raises(InvalidBackboneOutputError):
        BackboneValidator.validate(output)


def test_nan_tensor():
    output = create_backbone_output()

    output.stage1.tensor[:] = torch.nan

    with pytest.raises(InvalidFeatureMapError):
        BackboneValidator.validate(output)


def test_inf_tensor():
    output = create_backbone_output()

    output.stage2.tensor[:] = torch.inf

    with pytest.raises(InvalidFeatureMapError):
        BackboneValidator.validate(output)


def test_invalid_tensor_dimension():
    fmap = FeatureMap(
        stage="stage1",
        tensor=torch.randn(3, 224, 224),
        channels=3,
        stride=4,
    )

    output = BackboneOutput(
        stage1=fmap,
        stage2=create_feature_map(192, 28, 28, 8),
        stage3=create_feature_map(384, 14, 14, 16),
        stage4=create_feature_map(768, 7, 7, 32),
        original_size=(224, 224),
    )

    with pytest.raises(InvalidFeatureMapError):
        BackboneValidator.validate(output)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
def test_mixed_devices():
    stage1 = FeatureMap(
        stage="stage1",
        tensor=torch.randn(1, 96, 56, 56),
        channels=96,
        stride=4,
    )

    stage2 = FeatureMap(
        stage="stage2",
        tensor=torch.randn(
            1,
            192,
            28,
            28,
            device="cuda",
        ),
        channels=192,
        stride=8,
    )

    output = BackboneOutput(
        stage1=stage1,
        stage2=stage2,
        stage3=create_feature_map(384, 14, 14, 16),
        stage4=create_feature_map(768, 7, 7, 32),
        original_size=(224, 224),
    )

    with pytest.raises(InvalidFeatureMapError):
        BackboneValidator.validate(output)