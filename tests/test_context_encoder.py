"""
Unit tests for ContextEncoder.

Author: Ishita Dutta
Project: UIFAF
"""

import pytest
import torch

from attention.attention_models import (
    AttentionFeatureMap,
    AttentionOutput,
)

from representation.context_encoder import ContextEncoder
from representation.representation_models import (
    RepresentationFeature,
)
from representation.exceptions import (
    RepresentationInitializationError,
)


STAGE_CHANNELS = {
    "stage1": 96,
    "stage2": 192,
    "stage3": 384,
    "stage4": 768,
}


def make_feature(
    stage: str,
    channels: int,
    size: int,
    stride: int,
) -> AttentionFeatureMap:

    return AttentionFeatureMap(
        stage=stage,
        tensor=torch.randn(2, channels, size, size),
        channels=channels,
        stride=stride,
        attention_map=None,
        attention_name="EMA",
    )


def make_attention_output():

    return AttentionOutput(
        stage1=make_feature("stage1", 96, 56, 4),
        stage2=make_feature("stage2", 192, 28, 8),
        stage3=make_feature("stage3", 384, 14, 16),
        stage4=make_feature("stage4", 768, 7, 32),
        original_size=(224, 224),
    )


@pytest.fixture
def encoder():

    return ContextEncoder(STAGE_CHANNELS)


@pytest.fixture
def attention_output():

    return make_attention_output()


def test_initialization(encoder):

    assert isinstance(
        encoder,
        ContextEncoder,
    )

    assert len(encoder.blocks) == 4
    assert len(encoder.gates) == 4


def test_forward(encoder, attention_output):

    output = encoder(attention_output)

    assert isinstance(output, dict)

    assert len(output) == 4


def test_output_type(encoder, attention_output):

    output = encoder(attention_output)

    for feature in output.values():

        assert isinstance(
            feature,
            RepresentationFeature,
        )


def test_representation_type(encoder, attention_output):

    output = encoder(attention_output)

    for feature in output.values():

        assert (
            feature.representation_type
            == "context"
        )


def test_shape_preserved(
    encoder,
    attention_output,
):

    output = encoder(attention_output)

    assert (
        output["stage1"].tensor.shape
        == attention_output.stage1.tensor.shape
    )

    assert (
        output["stage2"].tensor.shape
        == attention_output.stage2.tensor.shape
    )

    assert (
        output["stage3"].tensor.shape
        == attention_output.stage3.tensor.shape
    )

    assert (
        output["stage4"].tensor.shape
        == attention_output.stage4.tensor.shape
    )


def test_invalid_stage_configuration():

    with pytest.raises(
        RepresentationInitializationError
    ):

        ContextEncoder(
            {
                "stage1": 96,
                "stage2": 192,
            }
        )


def test_invalid_tensor():

    encoder = ContextEncoder(STAGE_CHANNELS)

    with pytest.raises(Exception):

        encoder.blocks["stage1"](
            torch.randn(96, 56)
        )


def test_nan_detection(encoder, attention_output):

    attention_output.stage1.tensor[
        0, 0, 0, 0
    ] = float("nan")

    with pytest.raises(Exception):

        encoder(attention_output)