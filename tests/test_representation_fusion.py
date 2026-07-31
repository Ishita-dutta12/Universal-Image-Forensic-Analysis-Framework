import pytest
import torch

from representation.representation_fusion import RepresentationFusion
from representation.representation_models import (
    RepresentationFeature,
    StageRepresentation,
)
from representation.exceptions import (
    InvalidRepresentationFeatureError,
)

STAGE_CHANNELS = {
    "stage1": 96,
    "stage2": 192,
    "stage3": 384,
    "stage4": 768,
}


def make_feature(stage: str, channels: int, size: int):

    return RepresentationFeature(
        stage=stage,
        tensor=torch.randn(2, channels, size, size),
        representation_type="spatial",
    )


def make_stage(stage, channels, size):

    spatial = RepresentationFeature(
        stage=stage,
        tensor=torch.randn(2, channels, size, size),
        representation_type="spatial",
    )

    frequency = RepresentationFeature(
        stage=stage,
        tensor=torch.randn(2, channels, size, size),
        representation_type="frequency",
    )

    context = RepresentationFeature(
        stage=stage,
        tensor=torch.randn(2, channels, size, size),
        representation_type="context",
    )

    return spatial, frequency, context


@pytest.fixture
def fusion():

    return RepresentationFusion(STAGE_CHANNELS)


@pytest.fixture
def representations():

    spatial = {}
    frequency = {}
    context = {}

    sizes = {
        "stage1": 56,
        "stage2": 28,
        "stage3": 14,
        "stage4": 7,
    }

    for stage, ch in STAGE_CHANNELS.items():

        s, f, c = make_stage(stage, ch, sizes[stage])

        spatial[stage] = s
        frequency[stage] = f
        context[stage] = c

    return spatial, frequency, context


def test_initialization(fusion):

    assert isinstance(fusion, RepresentationFusion)


def test_forward(fusion, representations):

    spatial, frequency, context = representations

    output = fusion(
        spatial,
        frequency,
        context,
    )

    assert isinstance(output, dict)
    assert len(output) == 4


def test_stage_representation_type(fusion, representations):

    spatial, frequency, context = representations

    output = fusion(
        spatial,
        frequency,
        context,
    )

    for value in output.values():

        assert isinstance(
            value,
            StageRepresentation,
        )


def test_fused_representation_type(fusion, representations):

    spatial, frequency, context = representations

    output = fusion(
        spatial,
        frequency,
        context,
    )

    for value in output.values():

        assert value.fused.representation_type == "fused"


def test_shape_preserved(fusion, representations):

    spatial, frequency, context = representations

    output = fusion(
        spatial,
        frequency,
        context,
    )

    for stage in STAGE_CHANNELS:

        assert (
            output[stage].fused.tensor.shape
            ==
            spatial[stage].tensor.shape
        )


def test_invalid_stage_configuration():

    with pytest.raises(ValueError):

        RepresentationFusion(
            {
                "stage1": 96,
                "stage2": 192,
            }
        )


def test_stage_mismatch(fusion):

    spatial = RepresentationFeature(
        stage="stage1",
        tensor=torch.randn(2,96,56,56),
        representation_type="spatial",
    )

    frequency = RepresentationFeature(
        stage="stage2",
        tensor=torch.randn(2,96,56,56),
        representation_type="frequency",
    )

    context = RepresentationFeature(
        stage="stage1",
        tensor=torch.randn(2,96,56,56),
        representation_type="context",
    )

    with pytest.raises(
        InvalidRepresentationFeatureError
    ):

        fusion._fuse_stage(
            spatial,
            frequency,
            context,
        )


def test_invalid_shape(fusion):

    spatial = RepresentationFeature(
        stage="stage1",
        tensor=torch.randn(2,96,56,56),
        representation_type="spatial",
    )

    frequency = RepresentationFeature(
        stage="stage1",
        tensor=torch.randn(2,96,28,28),
        representation_type="frequency",
    )

    context = RepresentationFeature(
        stage="stage1",
        tensor=torch.randn(2,96,56,56),
        representation_type="context",
    )

    with pytest.raises(
        InvalidRepresentationFeatureError
    ):

        fusion._fuse_stage(
            spatial,
            frequency,
            context,
        )