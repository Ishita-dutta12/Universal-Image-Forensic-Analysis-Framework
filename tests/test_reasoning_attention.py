"""
Tests for ufre.reasoning_attention
"""

import pytest
import torch
import torch.nn as nn

from ufre.reasoning_attention import ReasoningAttention

from ufre.ufre_models import (
    ConfidenceEstimate,
    EvidenceFeature,
    EvidenceType,
    StageEvidence,
)

CHANNELS = 8


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


def make_evidence_feature(
    evidence_type=EvidenceType.TEXTURE,
):
    feature = EvidenceFeature(
        stage="stage1",
        evidence_type=evidence_type,
        tensor=make_tensor(),
        confidence=make_confidence(),
        metadata={},
    )

    feature.validate()

    return feature


def make_stage_evidence():

    spatial = make_evidence_feature(
        EvidenceType.TEXTURE,
    )

    frequency = make_evidence_feature(
        EvidenceType.FREQUENCY,
    )

    context = make_evidence_feature(
        EvidenceType.SEMANTIC,
    )

    fused = make_evidence_feature(
        EvidenceType.UNKNOWN,
    )

    stage = StageEvidence(
        stage="stage1",
        evidences=(
            spatial,
            frequency,
            context,
            fused,
        ),
        fused_tensor=make_tensor(),
        confidence=make_confidence(),
        metadata={},
    )

    stage.validate()

    return stage


# ==========================================================
# Constructor
# ==========================================================

class TestConstructor:

    def test_constructor(self):

        model = ReasoningAttention(CHANNELS)

        assert model.channels == CHANNELS
        assert model.num_heads == 8

    def test_attention_module(self):

        model = ReasoningAttention(CHANNELS)

        assert isinstance(
            model.attention,
            nn.MultiheadAttention,
        )

    def test_feed_forward_exists(self):

        model = ReasoningAttention(CHANNELS)

        assert isinstance(
            model.feed_forward,
            nn.Sequential,
        )


# ==========================================================
# refine_evidence()
# ==========================================================

class TestRefineEvidence:

    def test_returns_four_features(self):

        model = ReasoningAttention(CHANNELS)

        stage = make_stage_evidence()

        refined = model.refine_evidence(
            stage.evidences,
        )

        assert len(refined) == 4

    def test_returns_evidence_feature(self):

        model = ReasoningAttention(CHANNELS)

        refined = model.refine_evidence(
            make_stage_evidence().evidences,
        )

        for evidence in refined:

            assert isinstance(
                evidence,
                EvidenceFeature,
            )

    def test_stage_preserved(self):

        model = ReasoningAttention(CHANNELS)

        refined = model.refine_evidence(
            make_stage_evidence().evidences,
        )

        for evidence in refined:

            assert evidence.stage == "stage1"

    def test_tensor_shape_preserved(self):

        model = ReasoningAttention(CHANNELS)

        stage = make_stage_evidence()

        refined = model.refine_evidence(
            stage.evidences,
        )

        for old, new in zip(
            stage.evidences,
            refined,
        ):

            assert (
                old.tensor.shape
                == new.tensor.shape
            )

    def test_metadata_copied(self):

        model = ReasoningAttention(CHANNELS)

        stage = make_stage_evidence()

        refined = model.refine_evidence(
            stage.evidences,
        )

        for evidence in refined:

            assert isinstance(
                evidence.metadata,
                dict,
            )
    # ==========================================================
# reason_stage()
# ==========================================================

class TestReasonStage:

    def test_returns_stage_evidence(self):

        model = ReasoningAttention(CHANNELS)

        output = model.reason_stage(
            make_stage_evidence(),
        )

        assert isinstance(
            output,
            StageEvidence,
        )

    def test_stage_name_preserved(self):

        model = ReasoningAttention(CHANNELS)

        output = model.reason_stage(
            make_stage_evidence(),
        )

        assert output.stage == "stage1"

    def test_four_evidences(self):

        model = ReasoningAttention(CHANNELS)

        output = model.reason_stage(
            make_stage_evidence(),
        )

        assert len(output.evidences) == 4

    def test_attention_metadata_added(self):

        model = ReasoningAttention(CHANNELS)

        output = model.reason_stage(
            make_stage_evidence(),
        )

        assert output.metadata[
            "attention_refined"
        ] is True

    def test_fused_tensor_shape(self):

        model = ReasoningAttention(CHANNELS)

        stage = make_stage_evidence()

        output = model.reason_stage(stage)

        assert (
            output.fused_tensor.shape
            == stage.fused_tensor.shape
        )


# ==========================================================
# forward()
# ==========================================================

class TestForward:

    def test_forward_returns_tuple(self):

        model = ReasoningAttention(CHANNELS)

        stages = (
            make_stage_evidence(),
            make_stage_evidence(),
            make_stage_evidence(),
            make_stage_evidence(),
        )

        output = model(stages)

        assert isinstance(output, tuple)

    def test_forward_four_outputs(self):

        model = ReasoningAttention(CHANNELS)

        stages = (
            make_stage_evidence(),
            make_stage_evidence(),
            make_stage_evidence(),
            make_stage_evidence(),
        )

        output = model(stages)

        assert len(output) == 4

    def test_forward_stage_types(self):

        model = ReasoningAttention(CHANNELS)

        stages = (
            make_stage_evidence(),
            make_stage_evidence(),
            make_stage_evidence(),
            make_stage_evidence(),
        )

        output = model(stages)

        for stage in output:

            assert isinstance(
                stage,
                StageEvidence,
            )


# ==========================================================
# Helper methods
# ==========================================================

class TestHelpers:

    def test_num_heads(self):

        model = ReasoningAttention(
            CHANNELS,
            num_heads=4,
        )

        assert model.num_heads_used() == 4

    def test_embedding_dimension(self):

        model = ReasoningAttention(CHANNELS)

        assert (
            model.embedding_dimension()
            == CHANNELS
        )

    def test_supports_stage_true(self):

        model = ReasoningAttention(CHANNELS)

        assert model.supports_stage(
            make_stage_evidence()
        )

    def test_supports_stage_wrong_channels(self):

        model = ReasoningAttention(CHANNELS)

        stage = make_stage_evidence()

        stage.fused_tensor = torch.randn(
            1,
            16,
            16,
            16,
        )

        assert (
            model.supports_stage(stage)
            is False
        )

    def test_extra_repr(self):

        model = ReasoningAttention(
            CHANNELS,
            num_heads=4,
        )

        text = model.extra_repr()

        assert "channels=8" in text
        assert "heads=4" in text