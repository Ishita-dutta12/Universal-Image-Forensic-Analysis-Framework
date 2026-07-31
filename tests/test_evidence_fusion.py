"""
Tests for ufre.evidence_fusion
"""

import torch
import torch.nn as nn

from ufre.evidence_fusion import EvidenceFusion

from ufre.ufre_models import (
    ConfidenceEstimate,
    EvidenceFeature,
    EvidenceType,
    StageEvidence,
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

    stage = StageEvidence(
        stage="stage1",
        evidences=(
            make_evidence_feature(EvidenceType.TEXTURE),
            make_evidence_feature(EvidenceType.FREQUENCY),
            make_evidence_feature(EvidenceType.SEMANTIC),
            make_evidence_feature(EvidenceType.UNKNOWN),
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

        fusion = EvidenceFusion(
            CHANNELS,
            EMBED_DIM,
        )

        assert fusion.channels == CHANNELS
        assert fusion.embedding_dim == EMBED_DIM
        assert fusion.num_heads == 8

    def test_attention_exists(self):

        fusion = EvidenceFusion(
            CHANNELS,
            EMBED_DIM,
        )

        assert isinstance(
            fusion.stage_attention,
            nn.MultiheadAttention,
        )

    def test_projection_exists(self):

        fusion = EvidenceFusion(
            CHANNELS,
            EMBED_DIM,
        )

        assert isinstance(
            fusion.projection,
            nn.Sequential,
        )


# ==========================================================
# encode_stage()
# ==========================================================

class TestEncodeStage:

    def test_returns_tensor(self):

        fusion = EvidenceFusion(
            CHANNELS,
            EMBED_DIM,
        )

        token = fusion.encode_stage(
            make_stage_evidence(),
        )

        assert isinstance(
            token,
            torch.Tensor,
        )

    def test_correct_shape(self):

        fusion = EvidenceFusion(
            CHANNELS,
            EMBED_DIM,
        )

        token = fusion.encode_stage(
            make_stage_evidence(),
        )

        assert token.shape == (
            1,
            CHANNELS,
        )

    def test_preserves_dtype(self):

        fusion = EvidenceFusion(
            CHANNELS,
            EMBED_DIM,
        )

        stage = make_stage_evidence()

        token = fusion.encode_stage(stage)

        assert (
            token.dtype
            == stage.fused_tensor.dtype
        )

    def test_preserves_device(self):

        fusion = EvidenceFusion(
            CHANNELS,
            EMBED_DIM,
        )

        stage = make_stage_evidence()

        token = fusion.encode_stage(stage)

        assert (
            token.device
            == stage.fused_tensor.device
        )
    # ==========================================================
# forward()
# ==========================================================

class TestForward:

    def test_forward_returns_tensor(self):

        fusion = EvidenceFusion(
            CHANNELS,
            EMBED_DIM,
        )

        embedding = fusion(
            (
                make_stage_evidence(),
                make_stage_evidence(),
                make_stage_evidence(),
                make_stage_evidence(),
            )
        )

        assert isinstance(
            embedding,
            torch.Tensor,
        )

    def test_embedding_shape(self):

        fusion = EvidenceFusion(
            CHANNELS,
            EMBED_DIM,
        )

        embedding = fusion(
            (
                make_stage_evidence(),
                make_stage_evidence(),
                make_stage_evidence(),
                make_stage_evidence(),
            )
        )

        assert embedding.shape == (
            1,
            EMBED_DIM,
        )

    def test_embedding_dtype(self):

        fusion = EvidenceFusion(
            CHANNELS,
            EMBED_DIM,
        )

        embedding = fusion(
            (
                make_stage_evidence(),
                make_stage_evidence(),
                make_stage_evidence(),
                make_stage_evidence(),
            )
        )

        assert embedding.dtype == torch.float32


# ==========================================================
# supports_stage()
# ==========================================================

class TestSupportsStage:

    def test_supports_valid_stage(self):

        fusion = EvidenceFusion(
            CHANNELS,
            EMBED_DIM,
        )

        assert fusion.supports_stage(
            make_stage_evidence()
        )

    def test_wrong_channels(self):

        fusion = EvidenceFusion(
            CHANNELS,
            EMBED_DIM,
        )

        stage = make_stage_evidence()

        stage.fused_tensor = torch.randn(
            1,
            16,
            16,
            16,
        )

        assert (
            fusion.supports_stage(stage)
            is False
        )


# ==========================================================
# Helper methods
# ==========================================================

class TestHelpers:

    def test_embedding_size(self):

        fusion = EvidenceFusion(
            CHANNELS,
            EMBED_DIM,
        )

        assert (
            fusion.embedding_size()
            == EMBED_DIM
        )

    def test_num_attention_heads(self):

        fusion = EvidenceFusion(
            CHANNELS,
            EMBED_DIM,
            num_heads=4,
        )

        assert (
            fusion.num_attention_heads()
            == 4
        )

    def test_extra_repr(self):

        fusion = EvidenceFusion(
            CHANNELS,
            EMBED_DIM,
            num_heads=4,
        )

        text = fusion.extra_repr()

        assert "channels=8" in text
        assert "embedding_dim=64" in text
        assert "num_heads=4" in text