"""
Tests for ufre.confidence_estimator
"""

import pytest
import torch
import torch.nn as nn

from ufre.confidence_estimator import ConfidenceEstimator

from ufre.ufre_models import (
    ConfidenceEstimate,
)

EMBED_DIM = 64
HIDDEN_DIM = 32


# ==========================================================
# Builders
# ==========================================================

def make_embedding(
    batch_size=1,
    embedding_dim=EMBED_DIM,
):
    return torch.randn(
        batch_size,
        embedding_dim,
    )


# ==========================================================
# Constructor
# ==========================================================

class TestConstructor:

    def test_constructor(self):

        model = ConfidenceEstimator(
            EMBED_DIM,
            HIDDEN_DIM,
        )

        assert model.embedding_dim == EMBED_DIM
        assert model.hidden_dim == HIDDEN_DIM

    def test_confidence_head_exists(self):

        model = ConfidenceEstimator(
            EMBED_DIM,
            HIDDEN_DIM,
        )

        assert isinstance(
            model.confidence_head,
            nn.Sequential,
        )

    def test_uncertainty_head_exists(self):

        model = ConfidenceEstimator(
            EMBED_DIM,
            HIDDEN_DIM,
        )

        assert isinstance(
            model.uncertainty_head,
            nn.Sequential,
        )


# ==========================================================
# predict_confidence()
# ==========================================================

class TestPredictConfidence:

    def test_returns_tensor(self):

        model = ConfidenceEstimator(
            EMBED_DIM,
            HIDDEN_DIM,
        )

        output = model.predict_confidence(
            make_embedding(),
        )

        assert isinstance(
            output,
            torch.Tensor,
        )

    def test_correct_shape(self):

        model = ConfidenceEstimator(
            EMBED_DIM,
            HIDDEN_DIM,
        )

        output = model.predict_confidence(
            make_embedding(),
        )

        assert output.shape == (1, 1)

    def test_batch_shape(self):

        model = ConfidenceEstimator(
            EMBED_DIM,
            HIDDEN_DIM,
        )

        output = model.predict_confidence(
            make_embedding(batch_size=8),
        )

        assert output.shape == (
            8,
            1,
        )

    def test_invalid_dimension(self):

        model = ConfidenceEstimator(
            EMBED_DIM,
            HIDDEN_DIM,
        )

        with pytest.raises(ValueError):

            model.predict_confidence(
                torch.randn(
                    EMBED_DIM,
                )
            )


# ==========================================================
# predict_uncertainty()
# ==========================================================

class TestPredictUncertainty:

    def test_returns_tensor(self):

        model = ConfidenceEstimator(
            EMBED_DIM,
            HIDDEN_DIM,
        )

        output = model.predict_uncertainty(
            make_embedding(),
        )

        assert isinstance(
            output,
            torch.Tensor,
        )

    def test_correct_shape(self):

        model = ConfidenceEstimator(
            EMBED_DIM,
            HIDDEN_DIM,
        )

        output = model.predict_uncertainty(
            make_embedding(),
        )

        assert output.shape == (1, 1)

    def test_invalid_dimension(self):

        model = ConfidenceEstimator(
            EMBED_DIM,
            HIDDEN_DIM,
        )

        with pytest.raises(ValueError):

            model.predict_uncertainty(
                torch.randn(
                    EMBED_DIM,
                )
            )
            # ==========================================================
# forward()
# ==========================================================

class TestForward:

    def test_returns_confidence_estimate(self):

        model = ConfidenceEstimator(
            EMBED_DIM,
            HIDDEN_DIM,
        )

        estimate = model(
            make_embedding(),
        )

        assert isinstance(
            estimate,
            ConfidenceEstimate,
        )

    def test_confidence_range(self):

        model = ConfidenceEstimator(
            EMBED_DIM,
            HIDDEN_DIM,
        )

        estimate = model(
            make_embedding(),
        )

        assert 0.0 <= estimate.confidence <= 1.0

    def test_uncertainty_range(self):

        model = ConfidenceEstimator(
            EMBED_DIM,
            HIDDEN_DIM,
        )

        estimate = model(
            make_embedding(),
        )

        assert 0.0 <= estimate.uncertainty <= 1.0

    def test_batch_forward(self):

        model = ConfidenceEstimator(
            EMBED_DIM,
            HIDDEN_DIM,
        )

        estimate = model(
            make_embedding(batch_size=8),
        )

        assert isinstance(
            estimate,
            ConfidenceEstimate,
        )

    def test_invalid_embedding_dimension(self):

        model = ConfidenceEstimator(
            EMBED_DIM,
            HIDDEN_DIM,
        )

        with pytest.raises(ValueError):

            model(
                torch.randn(
                    EMBED_DIM,
                )
            )


# ==========================================================
# Helper methods
# ==========================================================

class TestHelpers:

    def test_output_dimension(self):

        model = ConfidenceEstimator(
            EMBED_DIM,
            HIDDEN_DIM,
        )

        assert (
            model.output_dimension()
            == EMBED_DIM
        )

    def test_supports_batch(self):

        model = ConfidenceEstimator(
            EMBED_DIM,
            HIDDEN_DIM,
        )

        assert model.supports_batch() is True

    def test_extra_repr(self):

        model = ConfidenceEstimator(
            EMBED_DIM,
            HIDDEN_DIM,
        )

        text = model.extra_repr()

        assert "embedding_dim=64" in text
        assert "hidden_dim=32" in text