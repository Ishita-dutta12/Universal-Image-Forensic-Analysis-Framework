"""
Tests for Cross Representation Module.

This script validates:

1. CrossAttention
2. GatedInteraction
3. InteractionFusion
4. Complete parallel pipeline

Author: Ishita Dutta
Project: UIFAF
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import torch

from representation.representation_models import (
    RepresentationFeature,
    StageRepresentation,
    RepresentationOutput,
)

from cross_representation.cross_attention import (
    CrossAttention,
)

from cross_representation.gated_interaction import (
    GatedInteraction,
)

from cross_representation.interaction_fusion import (
    InteractionFusion,
)


def create_feature(
    stage: str,
    representation_type: str,
    channels: int,
    height: int,
    width: int,
) -> RepresentationFeature:

    feature = RepresentationFeature(
        stage=stage,
        representation_type=representation_type,
        tensor=torch.randn(
            2,
            channels,
            height,
            width,
            requires_grad=True,
        ),
    )

    feature.validate()

    return feature


def create_stage(
    stage: str,
    channels: int,
    height: int,
    width: int,
) -> StageRepresentation:

    output = StageRepresentation(
        stage=stage,

        spatial=create_feature(
            stage,
            "spatial",
            channels,
            height,
            width,
        ),

        frequency=create_feature(
            stage,
            "frequency",
            channels,
            height,
            width,
        ),

        context=create_feature(
            stage,
            "context",
            channels,
            height,
            width,
        ),

        fused=create_feature(
            stage,
            "fused",
            channels,
            height,
            width,
        ),
    )

    output.validate()

    return output


def create_representation() -> RepresentationOutput:

    output = RepresentationOutput(

        stage1=create_stage(
            "stage1",
            96,
            56,
            56,
        ),

        stage2=create_stage(
            "stage2",
            192,
            28,
            28,
        ),

        stage3=create_stage(
            "stage3",
            384,
            14,
            14,
        ),

        stage4=create_stage(
            "stage4",
            768,
            7,
            7,
        ),

        original_size=(224, 224),
    )

    output.validate()

    return output
def test_cross_attention():

    print("\nTesting CrossAttention...")

    representation = create_representation()

    model = CrossAttention()

    output = model(representation)

    output.validate()

    print("✓ CrossAttention Passed")

    assert output is not None
    assert isinstance(output, RepresentationOutput)


def test_gated_interaction():

    print("\nTesting GatedInteraction...")

    representation = create_representation()

    model = GatedInteraction()

    output = model(representation)

    output.validate()

    print("✓ GatedInteraction Passed")

    assert output is not None
    assert isinstance(output, RepresentationOutput)


def test_interaction_fusion():

    print("\nTesting InteractionFusion...")

    representation = create_representation()

    attention = CrossAttention()(representation)

    gated = GatedInteraction()(representation)

    fusion = InteractionFusion()

    output = fusion(
        attention,
        gated,
    )

    output.validate()

    print("✓ InteractionFusion Passed")

    assert output is not None
    assert isinstance(output, RepresentationOutput)


def test_parallel_pipeline():

    print("\nTesting Complete Parallel Pipeline...")

    representation = create_representation()

    cross = CrossAttention()

    gate = GatedInteraction()

    fusion = InteractionFusion()

    attention_output = cross(
        representation
    )

    gated_output = gate(
        representation
    )

    final_output = fusion(
        attention_output,
        gated_output,
    )

    final_output.validate()

    print("✓ Parallel Pipeline Passed")

    assert final_output is not None
    assert isinstance(final_output, RepresentationOutput)

def test_shape_preservation():

    print("\nChecking Shape Preservation...")

    original = create_representation()

    cross = CrossAttention()
    gate = GatedInteraction()
    fusion = InteractionFusion()

    attention = cross(original)
    gated = gate(original)
    final = fusion(attention, gated)

    for in_stage, out_stage in zip(
        original.stages,
        final.stages,
    ):

        for name in (
            "spatial",
            "frequency",
            "context",
            "fused",
        ):

            in_feature = getattr(in_stage, name)
            out_feature = getattr(out_stage, name)

            if (
                in_feature is None
                or out_feature is None
            ):
                continue

            assert (
                in_feature.tensor.shape
                ==
                out_feature.tensor.shape
            )

    print("✓ Shapes Preserved")


def test_gradient():

    print("\nChecking Gradient Flow...")

    representation = create_representation()

    cross = CrossAttention()
    gate = GatedInteraction()
    fusion = InteractionFusion()

    attention = cross(representation)
    gated = gate(representation)
    final = fusion(attention, gated)

    loss = final.stage4.fused.tensor.mean()

    loss.backward()

    assert (
        final.stage4.fused.tensor.grad_fn
        is not None
    )

    print("✓ Gradient Flow Passed")