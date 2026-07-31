"""
Tests for Spatial Encoder.

Author: Ishita Dutta
Project: UIFAF
"""

import pytest
import torch

from attention.attention_models import (
    AttentionFeatureMap,
    AttentionOutput,
)

from representation.spatial_encoder import (
    SpatialBlock,
    SpatialEncoder,
)

from representation.exceptions import (
    RepresentationInitializationError,
)


def make_feature(stage: str, channels: int):

    stride_map = {
        "stage1": 4,
        "stage2": 8,
        "stage3": 16,
        "stage4": 32,
    }

    return AttentionFeatureMap(
        stage=stage,
        tensor=torch.randn(2, channels, 32, 32),
        attention_map=torch.randn(2, 8, 32, 32),
        channels=channels,
        stride=stride_map[stage],
    )


def make_attention_output():

    return AttentionOutput(

        stage1=make_feature("stage1", 96),

        stage2=make_feature("stage2", 192),

        stage3=make_feature("stage3", 384),

        stage4=make_feature("stage4", 768),

        original_size=(512, 512),
    )


def test_spatial_block_output_shape():

    block = SpatialBlock(96)

    x = torch.randn(2, 96, 32, 32)

    y = block(x)

    assert y.shape == x.shape


def test_invalid_channels():

    with pytest.raises(
        RepresentationInitializationError
    ):

        SpatialBlock(0)


def test_encoder_forward():

    encoder = SpatialEncoder(
        {
            "stage1": 96,
            "stage2": 192,
            "stage3": 384,
            "stage4": 768,
        }
    )

    output = encoder(
        make_attention_output()
    )

    output.validate()

    assert (
        output.stage1.spatial.tensor.shape
        == (2, 96, 32, 32)
    )

    assert (
        output.stage2.spatial.tensor.shape
        == (2, 192, 32, 32)
    )

    assert (
        output.stage3.spatial.tensor.shape
        == (2, 384, 32, 32)
    )

    assert (
        output.stage4.spatial.tensor.shape
        == (2, 768, 32, 32)
    )


def test_representation_type():

    encoder = SpatialEncoder(
        {
            "stage1": 96,
            "stage2": 192,
            "stage3": 384,
            "stage4": 768,
        }
    )

    output = encoder(
        make_attention_output()
    )

    assert (
        output.stage1.spatial.representation_type
        == "spatial"
    )

    assert (
        output.stage2.spatial.representation_type
        == "spatial"
    )

    assert (
        output.stage3.spatial.representation_type
        == "spatial"
    )

    assert (
        output.stage4.spatial.representation_type
        == "spatial"
    )


def test_original_size_preserved():

    encoder = SpatialEncoder(
        {
            "stage1": 96,
            "stage2": 192,
            "stage3": 384,
            "stage4": 768,
        }
    )

    output = encoder(
        make_attention_output()
    )

    assert output.original_size == (
        512,
        512,
    )