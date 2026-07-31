"""
End-to-end integration tests for UFRE.
"""

import torch

from representation.representation_models import (
    RepresentationFeature,
    StageRepresentation,
    RepresentationOutput,
)

from ufre.forensic_reasoner import ForensicReasoner
from ufre.ufre_models import UFREOutput


CHANNELS = 8
EMBED_DIM = 64


# ==========================================================
# Builders
# ==========================================================

def make_tensor(shape=(1, CHANNELS, 16, 16)):
    return torch.randn(*shape)


def make_feature(stage, representation):

    feature = RepresentationFeature(
        stage=stage,
        tensor=make_tensor(),
        representation_type=representation,
    )

    feature.validate()

    return feature


def make_stage(stage_name):

    stage = StageRepresentation(
        stage=stage_name,
        spatial=make_feature(stage_name, "spatial"),
        frequency=make_feature(stage_name, "frequency"),
        context=make_feature(stage_name, "context"),
        fused=make_feature(stage_name, "fused"),
    )

    stage.validate()

    return stage


def make_representation():

    output = RepresentationOutput(
        stage1=make_stage("stage1"),
        stage2=make_stage("stage2"),
        stage3=make_stage("stage3"),
        stage4=make_stage("stage4"),
        original_size=(256, 256),
    )

    output.validate()

    return output


# ==========================================================
# Integration Tests
# ==========================================================

class TestUFREIntegration:

    def test_complete_pipeline(self):

        model = ForensicReasoner(
            CHANNELS,
            EMBED_DIM,
        )

        output = model(
            make_representation(),
        )

        assert isinstance(
            output,
            UFREOutput,
        )

    def test_embedding_shape(self):

        model = ForensicReasoner(
            CHANNELS,
            EMBED_DIM,
        )

        output = model(
            make_representation(),
        )

        assert output.forensic_embedding.shape == (
            1,
            EMBED_DIM,
        )

    def test_four_stage_outputs(self):

        model = ForensicReasoner(
            CHANNELS,
            EMBED_DIM,
        )

        output = model(
            make_representation(),
        )

        assert output.stage1.stage == "stage1"
        assert output.stage2.stage == "stage2"
        assert output.stage3.stage == "stage3"
        assert output.stage4.stage == "stage4"

    def test_global_confidence(self):

        model = ForensicReasoner(
            CHANNELS,
            EMBED_DIM,
        )

        output = model(
            make_representation(),
        )

        assert (
            0.0
            <= output.global_confidence.confidence
            <= 1.0
        )

        assert (
            0.0
            <= output.global_confidence.uncertainty
            <= 1.0
        )

    def test_evidence_summary(self):

        model = ForensicReasoner(
            CHANNELS,
            EMBED_DIM,
        )

        output = model(
            make_representation(),
        )

        summary = output.evidence_summary

        assert summary.total_evidence == 16

        assert (
            0.0
            <= summary.average_confidence
            <= 1.0
        )

        assert summary.strongest_evidence is not None
        assert summary.weakest_evidence is not None

    def test_output_validation(self):

        model = ForensicReasoner(
            CHANNELS,
            EMBED_DIM,
        )

        output = model(
            make_representation(),
        )

        output.validate()