"""
Universal Forensic Reasoning Engine (UFRE) Data Models.

This module defines the core data structures used by the Universal
Forensic Reasoning Engine. These models represent forensic evidence,
confidence estimates, stage-level reasoning outputs, and the final
UFRE output consumed by downstream localization, decision, and report
generation modules.

Author: UIFAF
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from torch import Tensor


class EvidenceType(str, Enum):
    """
    Supported forensic evidence categories.

    Additional evidence types may be introduced in future versions
    without changing the public UFRE API.
    """

    NOISE = "noise"
    FREQUENCY = "frequency"
    CAMERA = "camera"
    COMPRESSION = "compression"
    METADATA = "metadata"
    TEXTURE = "texture"
    COLOR = "color"
    EDGE = "edge"
    SEMANTIC = "semantic"
    GAN_ARTIFACT = "gan_artifact"
    DIFFUSION_ARTIFACT = "diffusion_artifact"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class ConfidenceEstimate:
    """
    Represents the confidence and uncertainty associated with a
    forensic reasoning result.
    """

    confidence: float
    uncertainty: float

    def validate(self) -> None:
        """
        Validate confidence values.

        Raises
        ------
        ValueError
            If confidence or uncertainty is outside [0, 1].
        """

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be in [0, 1], got {self.confidence}."
            )

        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError(
                f"Uncertainty must be in [0, 1], got {self.uncertainty}."
            )

    @property
    def reliability(self) -> float:
        """
        Reliability score derived from uncertainty.

        Returns
        -------
        float
            Reliability in [0, 1].
        """
        return 1.0 - self.uncertainty
@dataclass(slots=True)
class EvidenceFeature:
    """
    Represents a single forensic evidence extracted from a stage
    representation.

    Parameters
    ----------
    stage : str
        Backbone stage that produced the evidence.
    evidence_type : EvidenceType
        Category of forensic evidence.
    tensor : Tensor
        Feature tensor describing the evidence.
    confidence : ConfidenceEstimate
        Confidence associated with this evidence.
    metadata : dict[str, Any]
        Optional metadata describing the evidence.
    """

    stage: str
    evidence_type: EvidenceType
    tensor: Tensor
    confidence: ConfidenceEstimate
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """
        Validate the evidence feature.

        Raises
        ------
        TypeError
            If an attribute has an invalid type.
        ValueError
            If an attribute contains invalid values.
        """

        if not isinstance(self.stage, str):
            raise TypeError(
                f"stage must be str, got {type(self.stage).__name__}."
            )

        if not self.stage:
            raise ValueError("stage cannot be empty.")

        if not isinstance(self.evidence_type, EvidenceType):
            raise TypeError(
                "evidence_type must be an instance of EvidenceType."
            )

        if not isinstance(self.tensor, Tensor):
            raise TypeError(
                f"tensor must be torch.Tensor, got {type(self.tensor).__name__}."
            )

        if self.tensor.numel() == 0:
            raise ValueError("Evidence tensor cannot be empty.")

        if not torch.isfinite(self.tensor).all():
            raise ValueError(
                "Evidence tensor contains NaN or Inf values."
            )

        self.confidence.validate()

        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary.")

    @property
    def device(self) -> torch.device:
        """Return the tensor device."""
        return self.tensor.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the tensor dtype."""
        return self.tensor.dtype

    @property
    def shape(self) -> torch.Size:
        """Return the tensor shape."""
        return self.tensor.shape

    @property
    def channels(self) -> int:
        """
        Return the channel dimension.

        Returns
        -------
        int
            Channel dimension.

        Raises
        ------
        ValueError
            If tensor has fewer than two dimensions.
        """
        if self.tensor.ndim < 2:
            raise ValueError(
                "Tensor must have at least 2 dimensions."
            )
        return self.tensor.shape[1]
@dataclass(slots=True)
class StageEvidence:
    """
    Represents the complete forensic evidence extracted from a single
    backbone stage.
    """

    stage: str
    evidences: tuple[EvidenceFeature, ...]
    fused_tensor: Tensor
    confidence: ConfidenceEstimate
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate the stage evidence."""

        if not isinstance(self.stage, str):
            raise TypeError(
                f"stage must be str, got {type(self.stage).__name__}."
            )

        if not self.stage:
            raise ValueError("stage cannot be empty.")

        if not isinstance(self.evidences, tuple):
            raise TypeError("evidences must be a tuple.")

        if len(self.evidences) == 0:
            raise ValueError(
                "StageEvidence must contain at least one EvidenceFeature."
            )

        for evidence in self.evidences:
            if not isinstance(evidence, EvidenceFeature):
                raise TypeError(
                    "All evidences must be EvidenceFeature instances."
                )
            evidence.validate()

        if not isinstance(self.fused_tensor, Tensor):
            raise TypeError(
                f"fused_tensor must be Tensor, got {type(self.fused_tensor).__name__}."
            )

        if self.fused_tensor.numel() == 0:
            raise ValueError("fused_tensor cannot be empty.")

        if not torch.isfinite(self.fused_tensor).all():
            raise ValueError(
                "fused_tensor contains NaN or Inf values."
            )

        self.confidence.validate()

        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary.")

    @property
    def num_evidences(self) -> int:
        """Return the number of evidence objects."""
        return len(self.evidences)

    @property
    def evidence_types(self) -> tuple[EvidenceType, ...]:
        """Return all evidence types."""
        return tuple(
            evidence.evidence_type
            for evidence in self.evidences
        )

    @property
    def device(self) -> torch.device:
        """Return tensor device."""
        return self.fused_tensor.device

    @property
    def dtype(self) -> torch.dtype:
        """Return tensor dtype."""
        return self.fused_tensor.dtype

    @property
    def shape(self) -> torch.Size:
        """Return tensor shape."""
        return self.fused_tensor.shape


@dataclass(slots=True)
class EvidenceSummary:
    """
    High-level summary of all forensic evidence produced by UFRE.
    """

    total_evidence: int
    average_confidence: float
    strongest_evidence: EvidenceType
    weakest_evidence: EvidenceType
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate summary statistics."""

        if self.total_evidence < 0:
            raise ValueError(
                "total_evidence cannot be negative."
            )

        if not 0.0 <= self.average_confidence <= 1.0:
            raise ValueError(
                "average_confidence must be in [0, 1]."
            )

        if not isinstance(
            self.strongest_evidence,
            EvidenceType,
        ):
            raise TypeError(
                "strongest_evidence must be EvidenceType."
            )

        if not isinstance(
            self.weakest_evidence,
            EvidenceType,
        ):
            raise TypeError(
                "weakest_evidence must be EvidenceType."
            )

        if not isinstance(self.metadata, dict):
            raise TypeError(
                "metadata must be a dictionary."
            )
@dataclass(slots=True)
class UFREOutput:
    """
    Final output of the Universal Forensic Reasoning Engine.

    This object aggregates all stage-level forensic evidence into a
    single representation for downstream localization, decision, and
    report generation modules.
    """

    stage1: StageEvidence
    stage2: StageEvidence
    stage3: StageEvidence
    stage4: StageEvidence

    forensic_embedding: Tensor

    global_confidence: ConfidenceEstimate

    evidence_summary: EvidenceSummary

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """
        Validate the complete UFRE output.

        Raises
        ------
        TypeError
            If any component has an invalid type.

        ValueError
            If tensors or confidence values are invalid.
        """

        for stage in self.stages:

            if not isinstance(stage, StageEvidence):
                raise TypeError(
                    f"Expected StageEvidence, got {type(stage).__name__}."
                )

            stage.validate()

        if not isinstance(self.forensic_embedding, Tensor):
            raise TypeError(
                "forensic_embedding must be a torch.Tensor."
            )

        if self.forensic_embedding.numel() == 0:
            raise ValueError(
                "forensic_embedding cannot be empty."
            )

        if not torch.isfinite(self.forensic_embedding).all():
            raise ValueError(
                "forensic_embedding contains NaN or Inf."
            )

        self.global_confidence.validate()

        self.evidence_summary.validate()

        if not isinstance(self.metadata, dict):
            raise TypeError(
                "metadata must be a dictionary."
            )

    @property
    def stages(
        self,
    ) -> tuple[
        StageEvidence,
        StageEvidence,
        StageEvidence,
        StageEvidence,
    ]:
        """
        Return all stage evidence.

        Returns
        -------
        tuple
            Stage1 through Stage4.
        """
        return (
            self.stage1,
            self.stage2,
            self.stage3,
            self.stage4,
        )

    @property
    def stage_names(self) -> tuple[str, str, str, str]:
        """
        Return stage names.
        """
        return (
            "stage1",
            "stage2",
            "stage3",
            "stage4",
        )

    @property
    def num_stages(self) -> int:
        """
        Number of backbone stages.
        """
        return 4

    @property
    def device(self) -> torch.device:
        """
        Device of the forensic embedding.
        """
        return self.forensic_embedding.device

    @property
    def dtype(self) -> torch.dtype:
        """
        Data type of the forensic embedding.
        """
        return self.forensic_embedding.dtype

    @property
    def shape(self) -> torch.Size:
        """
        Shape of the forensic embedding.
        """
        return self.forensic_embedding.shape