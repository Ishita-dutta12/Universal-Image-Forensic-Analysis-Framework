"""
Tests for ufre.forensic_reasoner
"""

import torch
import torch.nn as nn

from representation.representation_models import (
    RepresentationFeature,
    StageRepresentation,
    RepresentationOutput,
)

from ufre.forensic_reasoner import ForensicReasoner

from ufre.ufre_models import (
    ConfidenceEstimate,
    UFREOutput,
)

CHANNELS = 8
EMBED_DIM = 64


# ==========================================================
# Builders
# ==========================================================

def make_tensor(shape=(1, CHANNELS, 16, 16)):
    return torch.randn(*shape)


def make_confidence():
    return ConfidenceEstimate(
        confidence=0.9,
        uncertainty=0.1,
    )


def make_representation_feature(
    stage="stage1",
    representation_type="spatial",
):

    feature = RepresentationFeature(
        stage=stage,
        tensor=make_tensor(),
        representation_type=representation_type,
    )

    feature.validate()

    return feature


def make_stage(stage_name):

    stage = StageRepresentation(
        stage=stage_name,
        spatial=make_representation_feature(
            stage_name,
            "spatial",
        ),
        frequency=make_representation_feature(
            stage_name,
            "frequency",
        ),
        context=make_representation_feature(
            stage_name,
            "context",
        ),
        fused=make_representation_feature(
            stage_name,
            "fused",
        ),
    )

    stage.validate()

    return stage


def make_representation_output():

    output = RepresentationOutput(
        stage1=make_stage("stage1"),
        stage2=make_stage("stage2"),
        stage3=make_stage("stage3"),
        stage4=make_stage("stage4"),
        original_size=(256, 256),   # <-- add this
    )

    output.validate()

    return output


# ==========================================================
# Constructor
# ==========================================================

class TestConstructor:

    def test_constructor(self):

        model = ForensicReasoner(
            CHANNELS,
            EMBED_DIM,
        )

        assert model.embedding_dimension() == EMBED_DIM
        assert model.num_attention_heads() == 8

    def test_modules_created(self):

        model = ForensicReasoner(
            CHANNELS,
            EMBED_DIM,
        )

        assert model.encoder is not None
        assert model.reasoning is not None
        assert model.fusion is not None
        assert model.confidence is not None


# ==========================================================
# forward()
# ==========================================================

class TestForward:

    def test_returns_output(self):

        model = ForensicReasoner(
            CHANNELS,
            EMBED_DIM,
        )

        output = model(
            make_representation_output(),
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
            make_representation_output(),
        )

        assert output.forensic_embedding.shape == (
            1,
            EMBED_DIM,
        )

    def test_confidence_created(self):

        model = ForensicReasoner(
            CHANNELS,
            EMBED_DIM,
        )

        output = model(
            make_representation_output(),
        )

        assert output.global_confidence is not None

    def test_summary_created(self):

        model = ForensicReasoner(
            CHANNELS,
            EMBED_DIM,
        )

        output = model(
            make_representation_output(),
        )

        assert output.evidence_summary is not None
    # ==========================================================
# Evidence Summary
# ==========================================================

class TestEvidenceSummary:

    def test_total_evidence(self):

        model = ForensicReasoner(
            CHANNELS,
            EMBED_DIM,
        )

        output = model(
            make_representation_output(),
        )

        assert (
            output.evidence_summary.total_evidence
            == 16
        )

    def test_average_confidence_range(self):

        model = ForensicReasoner(
            CHANNELS,
            EMBED_DIM,
        )

        output = model(
            make_representation_output(),
        )

        assert (
            0.0
            <= output.evidence_summary.average_confidence
            <= 1.0
        )

    def test_strongest_evidence_exists(self):

        model = ForensicReasoner(
            CHANNELS,
            EMBED_DIM,
        )

        output = model(
            make_representation_output(),
        )

        assert (
            output.evidence_summary.strongest_evidence
            is not None
        )

    def test_weakest_evidence_exists(self):

        model = ForensicReasoner(
            CHANNELS,
            EMBED_DIM,
        )

        output = model(
            make_representation_output(),
        )

        assert (
            output.evidence_summary.weakest_evidence
            is not None
        )


# ==========================================================
# Stage Output
# ==========================================================

class TestStages:

    def test_stage_names(self):

        model = ForensicReasoner(
            CHANNELS,
            EMBED_DIM,
        )

        output = model(
            make_representation_output(),
        )

        assert output.stage1.stage == "stage1"
        assert output.stage2.stage == "stage2"
        assert output.stage3.stage == "stage3"
        assert output.stage4.stage == "stage4"

    def test_stage_objects_exist(self):

        model = ForensicReasoner(
            CHANNELS,
            EMBED_DIM,
        )

        output = model(
            make_representation_output(),
        )

        assert output.stage1 is not None
        assert output.stage2 is not None
        assert output.stage3 is not None
        assert output.stage4 is not None


# ==========================================================
# Helper methods
# ==========================================================

class TestHelpers:

    def test_embedding_dimension(self):

        model = ForensicReasoner(
            CHANNELS,
            EMBED_DIM,
        )

        assert (
            model.embedding_dimension()
            == EMBED_DIM
        )

    def test_num_attention_heads(self):

        model = ForensicReasoner(
            CHANNELS,
            EMBED_DIM,
            num_heads=4,
        )

        assert (
            model.num_attention_heads()
            == 4
        )

    def test_supports_batch(self):

        model = ForensicReasoner(
            CHANNELS,
            EMBED_DIM,
        )

        assert model.supports_batch() is True

    def test_extra_repr(self):

        model = ForensicReasoner(
            CHANNELS,
            EMBED_DIM,
            num_heads=4,
        )

        text = model.extra_repr()

        assert "embedding_dim=64" in text
        assert "attention_heads=4" in text