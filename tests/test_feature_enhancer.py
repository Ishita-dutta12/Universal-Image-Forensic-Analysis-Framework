
"""
Tests for FeatureEnhancer.

Author: Ishita Dutta
Project: UIFAF
"""

import torch

from attention.attention_models import AttentionOutput
from attention.feature_enhancer import FeatureEnhancer
from backbone.backbone_models import (
    BackboneOutput,
    FeatureMap,
)


def create_backbone_output():
    """
    Create a dummy BackboneOutput for testing.
    """

    return BackboneOutput(
        stage1=FeatureMap(
            stage="stage1",
            tensor=torch.randn(2, 96, 56, 56),
            channels=96,
            stride=4,
        ),
        stage2=FeatureMap(
            stage="stage2",
            tensor=torch.randn(2, 192, 28, 28),
            channels=192,
            stride=8,
        ),
        stage3=FeatureMap(
            stage="stage3",
            tensor=torch.randn(2, 384, 14, 14),
            channels=384,
            stride=16,
        ),
        stage4=FeatureMap(
            stage="stage4",
            tensor=torch.randn(2, 768, 7, 7),
            channels=768,
            stride=32,
        ),
        original_size=(224, 224),
    )


def test_initialization():
    enhancer = FeatureEnhancer()

    assert enhancer.stage1_attention.channels == 96
    assert enhancer.stage2_attention.channels == 192
    assert enhancer.stage3_attention.channels == 384
    assert enhancer.stage4_attention.channels == 768


def test_forward():
    enhancer = FeatureEnhancer()

    backbone = create_backbone_output()

    output = enhancer(backbone)

    assert isinstance(output, AttentionOutput)


def test_shapes_preserved():
    enhancer = FeatureEnhancer()

    backbone = create_backbone_output()

    output = enhancer(backbone)

    assert output.stage1.tensor.shape == backbone.stage1.tensor.shape
    assert output.stage2.tensor.shape == backbone.stage2.tensor.shape
    assert output.stage3.tensor.shape == backbone.stage3.tensor.shape
    assert output.stage4.tensor.shape == backbone.stage4.tensor.shape


def test_attention_maps_exist():
    enhancer = FeatureEnhancer()

    backbone = create_backbone_output()

    output = enhancer(backbone)

    assert output.stage1.attention_map is not None
    assert output.stage2.attention_map is not None
    assert output.stage3.attention_map is not None
    assert output.stage4.attention_map is not None


def test_original_size():
    enhancer = FeatureEnhancer()

    backbone = create_backbone_output()

    output = enhancer(backbone)

    assert output.original_size == (224, 224)


def test_finite_outputs():
    enhancer = FeatureEnhancer()

    backbone = create_backbone_output()

    output = enhancer(backbone)

    for stage in output.stages:
        assert torch.isfinite(stage.tensor).all()
        assert torch.isfinite(stage.attention_map).all()


def test_stage_names():
    enhancer = FeatureEnhancer()

    backbone = create_backbone_output()

    output = enhancer(backbone)

    assert output.stage_names == (
        "stage1",
        "stage2",
        "stage3",
        "stage4",
    )