"""
Confidence Estimator for UFRE.

Predicts confidence and uncertainty from the global forensic embedding.

Author: UIFAF
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ufre.ufre_models import (
    ConfidenceEstimate,
)


class ConfidenceEstimator(nn.Module):
    """
    Predict confidence and uncertainty from a forensic embedding.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.confidence_head = nn.Sequential(
            nn.Linear(
                embedding_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                1,
            ),
            nn.Sigmoid(),
        )

        self.uncertainty_head = nn.Sequential(
            nn.Linear(
                embedding_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                1,
            ),
            nn.Sigmoid(),
        )

    def predict_confidence(
        self,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict the confidence score.

        Parameters
        ----------
        embedding
            Global forensic embedding.

        Returns
        -------
        Tensor
            Confidence tensor of shape (B, 1).
        """

        if embedding.ndim != 2:
            raise ValueError(
                "Embedding must have shape (B, embedding_dim)."
            )

        return self.confidence_head(
            embedding
        )

    def predict_uncertainty(
        self,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict the uncertainty score.

        Parameters
        ----------
        embedding
            Global forensic embedding.

        Returns
        -------
        Tensor
            Uncertainty tensor of shape (B, 1).
        """

        if embedding.ndim != 2:
            raise ValueError(
                "Embedding must have shape (B, embedding_dim)."
            )

        return self.uncertainty_head(
            embedding
        )

    def forward(
        self,
        embedding: torch.Tensor,
    ) -> ConfidenceEstimate:
        """
        Estimate confidence and uncertainty from the forensic embedding.

        Parameters
        ----------
        embedding
            Global forensic embedding.

        Returns
        -------
        ConfidenceEstimate
            Estimated confidence and uncertainty.
        """

        if embedding.ndim != 2:
            raise ValueError(
                "Embedding must have shape (B, embedding_dim)."
            )

        confidence_tensor = self.predict_confidence(
            embedding
        )

        uncertainty_tensor = self.predict_uncertainty(
            embedding
        )

        estimate = ConfidenceEstimate(
            confidence=confidence_tensor.mean().item(),
            uncertainty=uncertainty_tensor.mean().item(),
        )

        estimate.validate()

        return estimate
    
    def output_dimension(
        self,
    ) -> int:
        """
        Return the expected embedding dimension.
        """

        return self.embedding_dim

    def supports_batch(
        self,
    ) -> bool:
        """
        Whether batch inference is supported.
        """

        return True

    def extra_repr(
        self,
    ) -> str:
        return (
            f"embedding_dim={self.embedding_dim}, "
            f"hidden_dim={self.hidden_dim}"
        )