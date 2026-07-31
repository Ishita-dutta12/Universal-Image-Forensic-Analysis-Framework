import pytest
import torch
import torch.nn as nn

from representation.representation_models import (
    RepresentationFeature,
    StageRepresentation,
    RepresentationOutput,
)

from ufre.evidence_encoder import EvidenceEncoder

from ufre.ufre_models import (
    ConfidenceEstimate,
    EvidenceFeature,
    EvidenceType,
    StageEvidence,
)


# ==========================================================
# Builders
# ==========================================================

CHANNELS = 8


def make_tensor():
    return torch.randn(1, CHANNELS, 16, 16)


def make_confidence():
    return ConfidenceEstimate(
        confidence=0.9,
        uncertainty=0.1,
    )


def make_representation_feature(
    stage="stage1",
    rep_type="spatial",
):
    return RepresentationFeature(
        stage=stage,
        tensor=make_tensor(),
        representation_type=rep_type,
    )


def make_stage_representation(stage="stage1"):
    return StageRepresentation(
        stage=stage,
        spatial=make_representation_feature(
            stage,
            "spatial",
        ),
        frequency=make_representation_feature(
            stage,
            "frequency",
        ),
        context=make_representation_feature(
            stage,
            "context",
        ),
        fused=make_representation_feature(
            stage,
            "fused",
        ),
    )


def make_representation_output():
    return RepresentationOutput(
        stage1=make_stage_representation("stage1"),
        stage2=make_stage_representation("stage2"),
        stage3=make_stage_representation("stage3"),
        stage4=make_stage_representation("stage4"),
        original_size=(256, 256),
    )


# ==========================================================
# Constructor
# ==========================================================

class TestConstructor:

    def test_constructor(self):

        encoder = EvidenceEncoder(CHANNELS)

        assert encoder.channels == CHANNELS

    def test_projection_count(self):

        encoder = EvidenceEncoder(CHANNELS)

        assert len(encoder.projections) == 4

    def test_projection_keys(self):

        encoder = EvidenceEncoder(CHANNELS)

        assert set(
            encoder.projections.keys()
        ) == {
            "spatial",
            "frequency",
            "context",
            "fused",
        }

    def test_projection_modules(self):

        encoder = EvidenceEncoder(CHANNELS)

        for module in encoder.projections.values():

            assert isinstance(
                module,
                nn.Conv2d,
            )
class TestEncodeFeature:

    def test_encode_spatial(self):

        encoder = EvidenceEncoder(CHANNELS)

        evidence = encoder.encode_feature(
            "stage1",
            "spatial",
            make_tensor(),
        )

        assert isinstance(
            evidence,
            EvidenceFeature,
        )

        assert evidence.stage == "stage1"

        assert (
            evidence.evidence_type
            == EvidenceType.TEXTURE
        )

    def test_encode_frequency(self):

        encoder = EvidenceEncoder(CHANNELS)

        evidence = encoder.encode_feature(
            "stage1",
            "frequency",
            make_tensor(),
        )

        assert (
            evidence.evidence_type
            == EvidenceType.FREQUENCY
        )

    def test_encode_context(self):

        encoder = EvidenceEncoder(CHANNELS)

        evidence = encoder.encode_feature(
            "stage1",
            "context",
            make_tensor(),
        )

        assert (
            evidence.evidence_type
            == EvidenceType.SEMANTIC
        )

    def test_encode_fused(self):

        encoder = EvidenceEncoder(CHANNELS)

        evidence = encoder.encode_feature(
            "stage1",
            "fused",
            make_tensor(),
        )

        assert (
            evidence.evidence_type
            == EvidenceType.UNKNOWN
        )