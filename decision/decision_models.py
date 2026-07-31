"""
UIFAF
Universal Image Forensic Analysis Framework

Module:
Decision Engine

File:
decision_models.py

Description:
Data models produced by the Decision Engine.

These models represent the final forensic decision before
report generation.

Author:
UIFAF Project
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from report_generator.report_models import (
    FinalVerdict,
)

from dataclasses import field
# ============================================================
# Image Classification
# ============================================================

class ImageClass(str, Enum):
    """
    Fine-grained forensic image classes.
    """

    AUTHENTIC = "Authentic"

    AI_GENERATED = "AI Generated"

    AI_EDITED = "AI Edited"

    FACE_SWAP = "Face Swap"

    DEEPFAKE = "Deepfake"

    COPY_MOVE = "Copy-Move Forgery"

    SPLICING = "Image Splicing"

    OBJECT_REMOVAL = "Object Removal"

    METADATA_TAMPERING = "Metadata Tampering"

    SCREENSHOT = "Screenshot / Re-photographed"

    CGI = "Computer Generated Image"

    UNKNOWN = "Unknown"


# ============================================================
# Classification Result
# ============================================================

@dataclass(slots=True, frozen=True)
class ClassificationResult:
    """
    Final image classification.
    """

    primary_class: ImageClass

    secondary_class: ImageClass | None

    probability: float


# ============================================================
# Evidence Score
# ============================================================

@dataclass(slots=True, frozen=True)
class EvidenceScore:
    """
    One forensic class probability.
    """

    image_class: ImageClass

    probability: float


# ============================================================
# Finding
# ============================================================

@dataclass(slots=True, frozen=True)
class FindingResult:
    """
    Human-readable forensic finding.
    """

    title: str

    description: str

    confidence: float

    source_module: str

    source_branch: str = ""

    supporting_evidence: list[str] = field(
        default_factory=list
    )


# ============================================================
# Decision Statistics
# ============================================================

@dataclass(slots=True, frozen=True)
class DecisionStatistics:
    """
    Statistics produced during decision making.
    """

    evidence_count: int

    average_confidence: float

    reasoning_confidence: float


# ============================================================
# Final Decision Output
# ============================================================

@dataclass(slots=True, frozen=True)
class DecisionOutput:
    """
    Final output of the Decision Engine.
    """

    classification: ClassificationResult

    final_verdict: FinalVerdict

    risk_score: float

    confidence: float

    uncertainty: float

    evidence_scores: list[EvidenceScore] = field(
        default_factory=list
    )

    findings: list[FindingResult] = field(
        default_factory=list
    )

    statistics: DecisionStatistics | None = None