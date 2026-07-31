"""
Tests for AttentionValidator.

Author: Ishita Dutta
Project: UIFAF
"""

from dataclasses import replace

import pytest
import torch

from attention.attention_models import (
    AttentionFeatureMap,
    AttentionOutput,
)
from attention.attention_validator import AttentionValidator
from attention.exceptions import InvalidAttentionOutputError


def create_attention_output():

    return AttentionOutput(
        stage1=AttentionFeatureMap(
            stage="stage1",
            tensor=torch.randn(2,96,56,56),
            channels=96,
            stride=4,
            attention_map=torch.randn(2,8,56,56),
        ),
        stage2=AttentionFeatureMap(
            stage="stage2",
            tensor=torch.randn(2,192,28,28),
            channels=192,
            stride=8,
            attention_map=torch.randn(2,8,28,28),
        ),
        stage3=AttentionFeatureMap(
            stage="stage3",
            tensor=torch.randn(2,384,14,14),
            channels=384,
            stride=16,
            attention_map=torch.randn(2,8,14,14),
        ),
        stage4=AttentionFeatureMap(
            stage="stage4",
            tensor=torch.randn(2,768,7,7),
            channels=768,
            stride=32,
            attention_map=torch.randn(2,8,7,7),
        ),
        original_size=(224,224),
    )


def test_validate():
    output = create_attention_output()

    AttentionValidator.validate(output)


def test_invalid_type():

    with pytest.raises(InvalidAttentionOutputError):
        AttentionValidator.validate(None)


def test_invalid_original_size():

    output = create_attention_output()

    output.original_size = (0,224)

    with pytest.raises(InvalidAttentionOutputError):
        AttentionValidator.validate(output)


def test_missing_attention_map():

    output = create_attention_output()

    output.stage1 = replace(
    output.stage1,
    attention_map=None,
)

    with pytest.raises(InvalidAttentionOutputError):
        AttentionValidator.validate(output)


def test_stage_order():

    output = create_attention_output()

    output.stage2 = replace(
    output.stage2,
    stage="wrong",
)

    with pytest.raises(InvalidAttentionOutputError):
        AttentionValidator.validate(output)


def test_nan_tensor():

    output = create_attention_output()

    new_tensor = output.stage3.tensor.clone()
    new_tensor[0, 0, 0, 0] = float("nan")

    output.stage3 = replace(
    output.stage3,
    tensor=new_tensor,
    )
    with pytest.raises(InvalidAttentionOutputError):
        AttentionValidator.validate(output)


def test_inf_attention():

    output = create_attention_output()

    new_attention = output.stage4.attention_map.clone()
    new_attention[0, 0, 0, 0] = float("inf")

    output.stage4 = replace(
    output.stage4,
    attention_map=new_attention,
    )

    with pytest.raises(InvalidAttentionOutputError):
        AttentionValidator.validate(output)
