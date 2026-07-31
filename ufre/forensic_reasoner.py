"""
Universal Forensic Reasoner (UFRE).

Main orchestration module for evidence encoding,
reasoning, fusion and confidence estimation.

Author: UIFAF
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from representation.representation_models import (
    RepresentationOutput,
)

from ufre.evidence_encoder import (
    EvidenceEncoder,
)

from ufre.reasoning_attention import (
    ReasoningAttention,
)

from ufre.evidence_fusion import (
    EvidenceFusion,
)

from ufre.confidence_estimator import (
    ConfidenceEstimator,
)

from ufre.ufre_models import (
    EvidenceSummary,
    EvidenceType,
    UFREOutput,
)
class ForensicReasoner(nn.Module):
    """
    Universal Forensic Reasoning Engine.
    """

    def __init__(
    self,
    stage_channels: dict[str, int],
    embedding_dim: int = 512,
    common_dim: int = 768,
    num_heads: int = 8,
) -> None:
        super().__init__()

        self.encoder = EvidenceEncoder(
            channels=stage_channels,
        )

        self.reasoning = ReasoningAttention(
            channels=stage_channels,
            num_heads=num_heads,
        )

        self.fusion = EvidenceFusion(
    stage_channels=stage_channels,
    common_dim=common_dim,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
)

        self.confidence = ConfidenceEstimator(
            embedding_dim=embedding_dim,
        )
        
    def forward(
        self,
        representation: RepresentationOutput,
    ) -> UFREOutput:
        """
        Run the complete UFRE pipeline.

        Parameters
        ----------
        representation
            Multi-stage representation features.

        Returns
        -------
        UFREOutput
            Final forensic reasoning output.
        """

        representation.validate()

        # Stage-wise evidence extraction
        stage_evidences = self.encoder(
            representation,
        )

        # Refine evidence using reasoning attention
        refined_evidences = self.reasoning(
            stage_evidences,
        )

        # Generate global forensic embedding
        forensic_embedding = self.fusion(
            refined_evidences,
        )
        print("Type:", type(forensic_embedding))

        import torch
        print("Is Tensor:", isinstance(forensic_embedding, torch.Tensor))

        # Estimate confidence
        global_confidence = self.confidence(
            forensic_embedding,
        )

        # Collect all evidence
        all_evidences = []

        for stage in refined_evidences:
            all_evidences.extend(
                stage.evidences
            )

        confidences = [
        evidence.confidence.confidence
        for evidence in all_evidences
        if evidence.confidence is not None
        ]

        average_confidence = (
            sum(confidences) / len(confidences)
            if confidences
            else 0.0
        )

        if all_evidences:
            strongest = max(
                all_evidences,
                key=lambda evidence: evidence.confidence.confidence,
            )

            weakest = min(
                all_evidences,
                key=lambda evidence: evidence.confidence.confidence,
            )

            strongest_type = strongest.evidence_type
            weakest_type = weakest.evidence_type

        else:
            strongest_type = weakest_type = EvidenceType.UNKNOWN

        evidence_summary = EvidenceSummary(
            total_evidence=len(all_evidences),
            average_confidence=average_confidence,
            strongest_evidence=strongest_type,
            weakest_evidence=weakest_type,
        )

        evidence_summary.validate()

        output = UFREOutput(
            stage1=refined_evidences[0],
            stage2=refined_evidences[1],
            stage3=refined_evidences[2],
            stage4=refined_evidences[3],
            forensic_embedding=forensic_embedding,
            global_confidence=global_confidence,
            evidence_summary=evidence_summary,
        )

        output.validate()

        return output
    def embedding_dimension(
        self,
    ) -> int:
        """
        Return the dimension of the forensic embedding.
        """

        return self.fusion.embedding_size()

    def num_attention_heads(
        self,
    ) -> int:
        """
        Return the number of attention heads used during reasoning.
        """

        return self.reasoning.num_heads_used()

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
            f"embedding_dim={self.embedding_dimension()}, "
            f"attention_heads={self.num_attention_heads()}"
        )  
        
