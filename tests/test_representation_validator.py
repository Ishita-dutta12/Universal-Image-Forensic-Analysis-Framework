import pytest
import torch

from representation.representation_validator import (
    RepresentationValidator,
)

from representation.representation_models import (
    RepresentationFeature,
    StageRepresentation,
    RepresentationOutput,
)

from representation.exceptions import (
    InvalidRepresentationFeatureError,
    InvalidRepresentationOutputError,
)


STAGE_CHANNELS = {
    "stage1": (96, 56),
    "stage2": (192, 28),
    "stage3": (384, 14),
    "stage4": (768, 7),
}


def make_feature(stage, channels, size, rep_type):

    return RepresentationFeature(
        stage=stage,
        tensor=torch.randn(2, channels, size, size),
        representation_type=rep_type,
    )


@pytest.fixture
def representation_output():

    stages = []

    for stage, (channels, size) in STAGE_CHANNELS.items():

        stages.append(
            StageRepresentation(
                stage=stage,
                spatial=make_feature(stage, channels, size, "spatial"),
                frequency=make_feature(stage, channels, size, "frequency"),
                context=make_feature(stage, channels, size, "context"),
                fused=make_feature(stage, channels, size, "fused"),
            )
        )

    return RepresentationOutput(
        stage1=stages[0],
        stage2=stages[1],
        stage3=stages[2],
        stage4=stages[3],
        original_size=(224, 224), 
    )


def test_validate(representation_output):

    RepresentationValidator.validate(
        representation_output
    )


def test_invalid_shape(representation_output):

    bad_frequency = RepresentationFeature(
    stage="stage1",
    tensor=torch.randn(2, 96, 28, 28),
    representation_type="frequency",
)

    bad_stage1 = StageRepresentation(
        stage="stage1",
        spatial=representation_output.stage1.spatial,
        frequency=bad_frequency,
        context=representation_output.stage1.context,
        fused=representation_output.stage1.fused,
    )

    representation_output = RepresentationOutput(
        stage1=bad_stage1,
        stage2=representation_output.stage2,
        stage3=representation_output.stage3,
        stage4=representation_output.stage4,
        original_size=representation_output.original_size,
    )

    with pytest.raises(
        InvalidRepresentationOutputError
    ):
        RepresentationValidator.validate(
            representation_output
        )


def test_invalid_batch(representation_output):

    bad_context = RepresentationFeature(
        stage="stage2",
        tensor=torch.randn(1, 192, 28, 28),
        representation_type="context",
)

    bad_stage2 = StageRepresentation(
        stage="stage2",
        spatial=representation_output.stage2.spatial,
        frequency=representation_output.stage2.frequency,
        context=bad_context,
        fused=representation_output.stage2.fused,
)

    representation_output = RepresentationOutput(
        stage1=representation_output.stage1,
        stage2=bad_stage2,
        stage3=representation_output.stage3,
        stage4=representation_output.stage4,
        original_size=representation_output.original_size,
)

    with pytest.raises(
        InvalidRepresentationOutputError
    ):
        RepresentationValidator.validate(
            representation_output
        )


def test_invalid_dtype(representation_output):

    bad_fused = RepresentationFeature(
        stage="stage3",
        tensor=representation_output.stage3.fused.tensor.double(),
        representation_type="fused",
)

    bad_stage3 = StageRepresentation(
        stage="stage3",
        spatial=representation_output.stage3.spatial,
        frequency=representation_output.stage3.frequency,
        context=representation_output.stage3.context,
        fused=bad_fused,
)

    representation_output = RepresentationOutput(
        stage1=representation_output.stage1,
        stage2=representation_output.stage2,
        stage3=bad_stage3,
        stage4=representation_output.stage4,
        original_size=representation_output.original_size,
)

    with pytest.raises(
        InvalidRepresentationOutputError
    ):
        RepresentationValidator.validate(
            representation_output
        )


def test_nan_detection(representation_output):

    representation_output.stage4.spatial.tensor[0, 0, 0, 0] = float("nan")

    with pytest.raises(
        InvalidRepresentationFeatureError
    ):
        RepresentationValidator.validate(
            representation_output
        )


def test_inf_detection(representation_output):

    representation_output.stage2.frequency.tensor[0, 0, 0, 0] = float("inf")

    with pytest.raises(
        InvalidRepresentationFeatureError
    ):
        RepresentationValidator.validate(
            representation_output
        )