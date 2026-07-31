"""
Evidence Encoder for UFRE.

Transforms refined backbone representations into structured forensic
evidence.

Author: UIFAF
"""

from __future__ import annotations

from scipy import spatial
import torch
import torch.nn as nn

from representation.representation_models import (
    StageRepresentation,
)

from ufre.base_reasoner import BaseReasoner

from ufre.ufre_models import (
    ConfidenceEstimate,
    EvidenceFeature,
    EvidenceType,
    StageEvidence,
)


class EvidenceEncoder(BaseReasoner):
    """
    Encodes refined representations into forensic evidence.
    """

    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        self.channels = channels

        self.projections = nn.ModuleDict({
    "stage1": nn.ModuleDict({
        "spatial": nn.Conv2d(96, 96, 1),
        "frequency": nn.Conv2d(96, 96, 1),
        "context": nn.Conv2d(96, 96, 1),
        "fused": nn.Conv2d(96, 96, 1),
    }),
    "stage2": nn.ModuleDict({
        "spatial": nn.Conv2d(192, 192, 1),
        "frequency": nn.Conv2d(192, 192, 1),
        "context": nn.Conv2d(192, 192, 1),
        "fused": nn.Conv2d(192, 192, 1),
    }),
    "stage3": nn.ModuleDict({
        "spatial": nn.Conv2d(384, 384, 1),
        "frequency": nn.Conv2d(384, 384, 1),
        "context": nn.Conv2d(384, 384, 1),
        "fused": nn.Conv2d(384, 384, 1),
    }),
    "stage4": nn.ModuleDict({
        "spatial": nn.Conv2d(768, 768, 1),
        "frequency": nn.Conv2d(768, 768, 1),
        "context": nn.Conv2d(768, 768, 1),
        "fused": nn.Conv2d(768, 768, 1),
    }),
})
            
        
    def encode_feature(
        self,
        stage: str,
        representation_type: str,
        tensor: torch.Tensor,
    ) -> EvidenceFeature:
       
        self.ensure_finite(tensor)

        if stage not in self.projections:
            
            raise ValueError(
        f"Unknown stage: {stage}"
    )

        if representation_type not in self.projections[stage]:
            raise ValueError(
                f"Unknown representation type: {representation_type}"
            )

        projected = self.projections[stage][representation_type](tensor)

        evidence_map = {
            
            "spatial": EvidenceType.TEXTURE,
            "frequency": EvidenceType.FREQUENCY,
            "context": EvidenceType.SEMANTIC,
            "fused": EvidenceType.UNKNOWN,
        }

        # Compute confidence from the projected feature tensor
        confidence_value = torch.sigmoid(
        projected.abs().mean()
        ).item()

        confidence = ConfidenceEstimate(
        confidence=confidence_value,
        uncertainty=1.0 - confidence_value,
)

        feature = EvidenceFeature(
            stage=stage,
            evidence_type=evidence_map[representation_type],
            tensor=projected,
            confidence=confidence,
            metadata={
                "source": representation_type,
            },
        )

        feature.validate()

        return feature
    
    def reason_stage(
        self,
        stage: StageRepresentation,
    ) -> StageEvidence:
        
        spatial = self.encode_feature(
            stage.stage,
            "spatial",
            stage.spatial.tensor,
        )

        frequency = self.encode_feature(
            stage.stage,
            "frequency",
            stage.frequency.tensor,
        )

        context = self.encode_feature(
            stage.stage,
            "context",
            stage.context.tensor,
        )

        fused = self.encode_feature(
            stage.stage,
            "fused",
            stage.fused.tensor,
        )

        fused_tensor = (
            spatial.tensor
            + frequency.tensor
            + context.tensor
            + fused.tensor
        ) / 4.0

        # Stage confidence = average of all evidence confidences
        stage_confidence = (
        spatial.confidence.confidence
        + frequency.confidence.confidence
        + context.confidence.confidence
        + fused.confidence.confidence
        ) / 4.0

        confidence = ConfidenceEstimate(
        confidence=stage_confidence,
            uncertainty=1.0 - stage_confidence,
)
        output = StageEvidence(
            stage=stage.stage,
            evidences=(
                spatial,
                frequency,
                context,
                fused,
            ),
            fused_tensor=fused_tensor,
            confidence=confidence,
            metadata={
                "num_evidences": 4,
            },
        )

        output.validate()

        return output
    def available_representations(
        self,
        ) -> tuple[str, ...]:
        return (
        "spatial",
        "frequency",
        "context",
        "fused",
    )

    def supports_representation(
        self,
        representation_type: str,
) -> bool:
        return representation_type in (
        "spatial",
        "frequency",
        "context",
        "fused",
    )

    def num_representations(self) -> int:
        """
        Return the number of supported representations.

        Returns
        -------
        int
        """
        return len(self.projections)

    def extra_repr(self) -> str:
        """
        String representation used by nn.Module.

        Returns
        -------
        str
        """
        return (
            f"channels={self.channels}, "
            f"representations={self.num_representations()}"
        ) 