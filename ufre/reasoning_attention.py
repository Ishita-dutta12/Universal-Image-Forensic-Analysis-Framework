"""
Reasoning Attention for UFRE.

Refines forensic evidence using multi-head self-attention.

Author: UIFAF
"""

from __future__ import annotations

import torch
import torch.nn as nn

#from ufre.base_reasoner import BaseReasoner

from ufre.ufre_models import (
    ConfidenceEstimate,
    EvidenceFeature,
    StageEvidence,
)


class ReasoningAttention(nn.Module):
    """
    Applies self-attention across evidence within a stage.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
    ) -> None:
        super().__init__()

        self.channels = channels
        self.num_heads = num_heads

        self.attention = nn.ModuleDict({
    "stage1": nn.MultiheadAttention(96, 8, batch_first=True),
    "stage2": nn.MultiheadAttention(192, 8, batch_first=True),
    "stage3": nn.MultiheadAttention(384, 8, batch_first=True),
    "stage4": nn.MultiheadAttention(768, 8, batch_first=True),
})

        self.norm1 = nn.ModuleDict({
    "stage1": nn.LayerNorm(96),
    "stage2": nn.LayerNorm(192),
    "stage3": nn.LayerNorm(384),
    "stage4": nn.LayerNorm(768),
})

        self.feed_forward = nn.ModuleDict({
    "stage1": nn.Sequential(
        nn.Linear(96, 384),
        nn.GELU(),
        nn.Linear(384, 96),
    ),
    "stage2": nn.Sequential(
        nn.Linear(192, 768),
        nn.GELU(),
        nn.Linear(768, 192),
    ),
    "stage3": nn.Sequential(
        nn.Linear(384, 1536),
        nn.GELU(),
        nn.Linear(1536, 384),
    ),
    "stage4": nn.Sequential(
        nn.Linear(768, 3072),
        nn.GELU(),
        nn.Linear(3072, 768),
    ),
})

        self.norm2 = nn.ModuleDict({
    "stage1": nn.LayerNorm(96),
    "stage2": nn.LayerNorm(192),
    "stage3": nn.LayerNorm(384),
    "stage4": nn.LayerNorm(768),
})
    def refine_evidence(
        self,
        evidences: tuple[
            EvidenceFeature,
            EvidenceFeature,
            EvidenceFeature,
            EvidenceFeature,
        ],
    ) -> tuple[
        EvidenceFeature,
        EvidenceFeature,
        EvidenceFeature,
        EvidenceFeature,
    ]:
        

        pooled = []

        for evidence in evidences:

            vector = (
                torch.nn.functional
                .adaptive_avg_pool2d(
                    evidence.tensor,
                    output_size=1,
                )
                .flatten(1)
            )

            pooled.append(vector)

        tokens = torch.stack(
            pooled,
            dim=1,
        )

        attended, _ = self.attention[
            evidences[0].stage
        ](
            tokens,
            tokens,
            tokens,
            )

        stage_name = evidences[0].stage

        tokens = self.norm1[stage_name](
            tokens + attended
            )

        ff = self.feed_forward[stage_name](
            tokens
        )

        tokens = self.norm2[stage_name](
            tokens + ff
        )

        refined = []

        for idx, evidence in enumerate(evidences):

            weight = (
                tokens[:, idx, :]
                .unsqueeze(-1)
                .unsqueeze(-1)
            )

            refined_tensor = (
                evidence.tensor
                + evidence.tensor * weight
                )

            refined.append(
                EvidenceFeature(
                    stage=evidence.stage,
                    evidence_type=evidence.evidence_type,
                    tensor=refined_tensor,
                    confidence=evidence.confidence,
                    metadata=dict(evidence.metadata),
                )
            )

        return tuple(refined)
    def reason_stage(
        self,
        stage: StageEvidence,
    ) -> StageEvidence:
        """
        Refine the forensic evidence for a single stage using
        self-attention.

        Parameters
        ----------
        stage
            StageEvidence produced by EvidenceEncoder.

        Returns
        -------
        StageEvidence
            Refined StageEvidence.
        """

        stage.validate()

        refined = self.refine_evidence(
            stage.evidences
        )

        fused_tensor = torch.stack(
            [e.tensor for e in refined],
            dim=0,
        ).mean(dim=0)

        confidence = ConfidenceEstimate(
            confidence=stage.confidence.confidence,
            uncertainty=stage.confidence.uncertainty,
        )

        output = StageEvidence(
            stage=stage.stage,
            evidences=refined,
            fused_tensor=fused_tensor,
            confidence=confidence,
            metadata={
                **stage.metadata,
                "attention_refined": True,
            },
        )

        output.validate()

        return output

    def forward(
        self,
        stages: tuple[
        StageEvidence,
        StageEvidence,
        StageEvidence,
        StageEvidence,
    ],
    ) -> tuple[
        StageEvidence,
        StageEvidence,
        StageEvidence,
        StageEvidence,
    ]:
    

        outputs = []

        for stage in stages:
            outputs.append(
                self.reason_stage(stage)
            )

        return tuple(outputs)
    def num_heads_used(self) -> int:
        """
        Return the number of attention heads.

        Returns
        -------
        int
            Number of attention heads.
        """
        return self.num_heads

    def embedding_dimension(self) -> int:
        """
        Return the attention embedding dimension.

        Returns
        -------
        int
            Embedding dimension.
        """
        return self.channels

    def supports_stage(
        self,
        stage: StageEvidence,
    ) -> bool:
        """
        Check whether a StageEvidence object is compatible
        with this attention module.

        Parameters
        ----------
        stage
            Stage evidence.

        Returns
        -------
        bool
        """

        if len(stage.evidences) != 4:
            return False

        channels = stage.fused_tensor.shape[1]

        return channels == self.channels

    def extra_repr(self) -> str:
        """
        String representation used by nn.Module.
        """

        return (
            f"channels={self.channels}, "
            f"heads={self.num_heads}"
        )