"""
Evidence Fusion module for UFRE.

Combines refined stage evidence into a unified forensic embedding.

Author: UIFAF
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ufre.ufre_models import (
    StageEvidence,
)


class EvidenceFusion(nn.Module):
    """
    Fuse multi-stage forensic evidence into a global embedding.
    """

    def __init__(
        self,
        stage_channels: dict[str, int],
        embedding_dim: int = 512,
        common_dim: int = 768,
        num_heads: int = 8,
    ):

        super().__init__()

        self.common_dim = common_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.stage_pool = nn.Linear(4, 1)
        self.stage_attention = nn.MultiheadAttention(
            embed_dim=common_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(common_dim)

        self.projection = nn.Sequential(
            nn.Linear(common_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(
                embedding_dim,
                embedding_dim,
            ),
        )

        self.stage_projection = nn.ModuleDict({
            "stage1": nn.Linear(stage_channels["stage1"], common_dim),
            "stage2": nn.Linear(stage_channels["stage2"], common_dim),
            "stage3": nn.Linear(stage_channels["stage3"], common_dim),
            "stage4": nn.Linear(stage_channels["stage4"], common_dim),
        })
    def encode_stage(
        self,
        stage: StageEvidence,
        stage_name: str,
    ) -> torch.Tensor:
        """
        Convert a StageEvidence object into a stage token.

        Parameters
        ----------
        stage
            Stage evidence.

        Returns
        -------
        Tensor
            Stage token of shape (B, C).
        """

        stage.validate()

        token = F.adaptive_avg_pool2d(
            stage.fused_tensor,
            output_size=1,
        )

        token = token.flatten(1)
        token = self.stage_projection[stage_name](token)

        return token
    def forward(
        self,
        stages: tuple[
            StageEvidence,
            StageEvidence,
            StageEvidence,
            StageEvidence,
        ],
    ) -> torch.Tensor:
        """
        Fuse four stages into one forensic embedding.

        Parameters
        ----------
        stages
            Four StageEvidence objects.

        Returns
        -------
        Tensor
            Global forensic embedding.
        """

        tokens = []

        for stage_name, stage in zip(self.stage_projection.keys(), stages):
            tokens.append(
                self.encode_stage(
                    stage,
                    stage_name,
                )
            )

        tokens = torch.stack(
            tokens,
            dim=1,
        )

        attended, _ = self.stage_attention(
            tokens,
            tokens,
            tokens,
        )

        tokens = self.norm(tokens + attended)

        # Compute soft attention weights over the 4 stage tokens.
        # tokens: (B, 4, C). Reduce across channels to get (B, 4, 1) scores.
        scores = torch.mean(tokens, dim=2, keepdim=True)
        weights = torch.softmax(scores, dim=1)  # (B, 4, 1)

        global_token = (tokens * weights).sum(dim=1)  # (B, C)

        embedding = self.projection(
            global_token
        )

        return embedding
    def supports_stage(
        self,
        stage: StageEvidence,
    ) -> bool:
        """
        Check whether a StageEvidence object is compatible
        with this fusion module.

        Parameters
        ----------
        stage
            Stage evidence.

        Returns
        -------
        bool
            True if compatible.
        """

        try:
            stage.validate()
        except Exception:
            return False

        return (
            stage.fused_tensor.ndim == 4
            and stage.fused_tensor.shape[1] == self.common_dim
        )

    def embedding_size(self) -> int:
        """
        Return the output embedding dimension.

        Returns
        -------
        int
        """
        return self.embedding_dim

    def num_attention_heads(self) -> int:
        """
        Return the number of attention heads.

        Returns
        -------
        int
        """
        return self.num_heads

    def extra_repr(self) -> str:
        """
        Additional module representation.

        Returns
        -------
        str
        """
        return (
            f"common_dim={self.common_dim}, "
            f"embedding_dim={self.embedding_dim}, "
            f"num_heads={self.num_heads}"
        )
