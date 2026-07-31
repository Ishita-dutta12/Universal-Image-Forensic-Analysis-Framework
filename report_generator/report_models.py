"""
UIFAF
Universal Image Forensic Analysis Framework

Module:
Universal Forensic Report Generator (UFRG)

File:
report_models.py

Description:
Defines immutable data models used by the Universal Forensic
Report Generator.

This file intentionally contains NO business logic.

Responsibilities
----------------
- Report data models
- Enums
- Type aliases
- Immutable report objects

Author:
UIFAF Project
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


# ============================================================
# Type Aliases
# ============================================================

ScoreMap = Dict[str, float]
MetadataMap = Dict[str, str]
ThresholdMap = Dict[str, float]


# ============================================================
# Enums
# ============================================================

class ConfidenceLevel(str, Enum):
    """Human readable confidence levels."""

    VERY_LOW = "Very Low"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    VERY_HIGH = "Very High"


class RiskLevel(str, Enum):
    """Overall manipulation risk."""

    MINIMAL = "Minimal"
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    CRITICAL = "Critical"


class FinalVerdict(str, Enum):
    """Overall forensic decision."""

    AUTHENTIC = "Authentic"
    MANIPULATED = "Manipulated"
    AI_GENERATED = "AI Generated"
    SUSPICIOUS = "Suspicious"
    UNKNOWN = "Unknown"


class RecommendationPriority(str, Enum):
    """Recommendation severity."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class RecommendationCategory(str, Enum):
    """Recommendation category."""

    GENERAL = "General"
    SECURITY = "Security"
    VERIFICATION = "Verification"
    FORENSICS = "Forensics"
    INVESTIGATION = "Investigation"


class ReportLevel(str, Enum):
    """Output verbosity."""

    BASIC = "Basic"
    DETAILED = "Detailed"
    RESEARCH = "Research"
class ImageClass(str, Enum):

    AUTHENTIC = "Authentic"

    AI_GENERATED = "AI Generated"

    AI_EDITED = "AI Edited"

    FACE_SWAP = "Face Swap"

    DEEPFAKE = "Deepfake"

    COPY_MOVE = "Copy Move"

    SPLICING = "Splicing"

    OBJECT_REMOVAL = "Object Removal"

    METADATA_TAMPERING = "Metadata Tampering"

    SCREENSHOT = "Screenshot"

    CGI = "CGI"

    UNKNOWN = "Unknown" 


# ============================================================
# File Information
# ============================================================

@dataclass(frozen=True)
class FileInformation:
    """
    Basic information about the analyzed image.
    """

    filename: str

    file_size_bytes: int

    width: int

    height: int

    image_format: str

    channels: int

    color_space: str

    resolution: str

    extension: str


# ============================================================
# Evidence Scores
# ============================================================

@dataclass(frozen=True)
class EvidenceScores:
    """
    Final normalized probability scores for every forensic class.

    Example
    -------
    {
        "AI Generated": 0.96,
        "Authenticity": 0.08,
        "Splicing": 0.04
    }
    """

    scores: dict[ImageClass, float] = field(
    default_factory=dict
)


# ============================================================
# Evidence Summary
# ============================================================

@dataclass(frozen=True)
class EvidenceSummary:
    """
    Summary of evidence confidence contributed by each branch.
    """

    total_evidence: int

    average_confidence: float

    strongest_evidence: str

    weakest_evidence: str

    overall_confidence: float
    # ============================================================
# Risk Assessment
# ============================================================

@dataclass(frozen=True)
class RiskAssessment:
    """
    Overall manipulation risk assessment.

    Attributes
    ----------
    risk_score:
        Overall manipulation probability (0.0 - 1.0).

    risk_level:
        Human readable risk category.
    """

    risk_score: float

    risk_level: RiskLevel


# ============================================================
# Confidence Assessment
# ============================================================

@dataclass(frozen=True)
class ConfidenceAssessment:
    """
    Confidence produced by the Confidence Estimator.
    """

    confidence: float

    uncertainty: float

    confidence_level: ConfidenceLevel


# ============================================================
# Finding
# ============================================================

@dataclass(frozen=True)
class Finding:
    """
    Represents one forensic finding produced by the framework.

    Every finding is completely traceable to its origin.
    """

    title: str

    description: str

    confidence: float

    source_module: str

    source_branch: str

    supporting_evidence: List[str] = field(default_factory=list)


# ============================================================
# Recommendation
# ============================================================

@dataclass(frozen=True)
class Recommendation:
    """
    One recommendation generated by the Recommendation Engine.
    """

    message: str

    priority: RecommendationPriority

    category: RecommendationCategory


# ============================================================
# Localization Result
# ============================================================

@dataclass(frozen=True)
class LocalizationResult:
    """
    Localization outputs produced by the framework.
    """

    heatmap_path: Optional[str] = None

    mask_path: Optional[str] = None

    bounding_box_path: Optional[str] = None

    manipulated_area_percentage: Optional[float] = None

    detected_regions: int = 0

    peak_confidence: Optional[float] = None
    
    # ============================================================
# Metadata Summary
# ============================================================

@dataclass(frozen=True)
class MetadataSummary:
    """
    Summary of image metadata extracted during preprocessing.

    Attributes
    ----------
    metadata:
        Extracted metadata key-value pairs.

    exif_present:
        Indicates whether EXIF metadata exists.

    metadata_modified:
        True if metadata appears to have been altered.

    suspicious_fields:
        List of suspicious metadata fields.

    software:
        Software used to create or modify the image.

    camera_model:
        Camera model, if available.

    gps_present:
        Whether GPS coordinates are available.
    """

    metadata: MetadataMap = field(default_factory=dict)

    exif_present: bool = False

    metadata_modified: bool = False

    suspicious_fields: List[str] = field(default_factory=list)

    software: Optional[str] = None

    camera_model: Optional[str] = None

    gps_present: bool = False


# ============================================================
# Raw Model Output
# ============================================================

@dataclass(frozen=True)
class RawModelOutput:
    """
    Raw prediction produced by a single model or branch.

    This object preserves intermediate outputs for
    debugging, explainability, and research.
    """

    module: str

    branch: str

    prediction: str

    probability: float

    confidence: float

    threshold: float


# ============================================================
# Raw Model Outputs
# ============================================================

@dataclass(frozen=True)
class RawModelOutputs:
    """
    Collection of all raw outputs produced during inference.

    Each pipeline stage may contribute one or more outputs.
    """

    outputs: List[RawModelOutput] = field(default_factory=list)


# ============================================================
# Threshold Information
# ============================================================

@dataclass(frozen=True)
class ThresholdInformation:
    """
    Thresholds used during final decision making.
    """

    thresholds: ThresholdMap = field(default_factory=dict)


# ============================================================
# System Information
# ============================================================

@dataclass(frozen=True)
class SystemInformation:
    """
    Runtime information about the forensic framework.
    """

    framework_name: str

    framework_version: str

    backbone: str

    attention_module: str

    reasoning_engine: str

    python_version: str

    device: str

    model_checkpoint: Optional[str] = None


# ============================================================
# Performance Statistics
# ============================================================

@dataclass(frozen=True)
class PerformanceStatistics:
    """
    Performance metrics collected during inference.
    """

    preprocessing_time_ms: float

    feature_extraction_time_ms: float

    reasoning_time_ms: float

    total_inference_time_ms: float

    memory_usage_mb: Optional[float] = None
    
    # ============================================================
# Master Report
# ============================================================

@dataclass(frozen=True)
class ForensicReport:
    """
    Master forensic report produced by UIFAF.

    This object is the single source of truth for every
    forensic analysis result. All output formats (JSON,
    API responses, dashboards, PDFs, etc.) should be
    generated from this object.
    """

    # ========================================================
    # Report Metadata
    # ========================================================

    report_id: str

    timestamp: datetime

    report_level: ReportLevel

    # ========================================================
    # Analysis Results
    # ========================================================

    file_information: FileInformation

    evidence_scores: EvidenceScores

    evidence_summary: EvidenceSummary

    risk_assessment: RiskAssessment

    confidence_assessment: ConfidenceAssessment

    final_verdict: FinalVerdict
    primary_class: ImageClass

    secondary_class: ImageClass | None = None

    classification_confidence: float = 0.0

    findings: List[Finding] = field(default_factory=list)

    recommendations: List[Recommendation] = field(default_factory=list)

    localization: Optional[LocalizationResult] = None

    metadata_summary: Optional[MetadataSummary] = None

    raw_outputs: Optional[RawModelOutputs] = None

    threshold_information: Optional[ThresholdInformation] = None

    system_information: Optional[SystemInformation] = None

    performance_statistics: Optional[PerformanceStatistics] = None

    # ========================================================
    # Convenience Properties
    # ========================================================

    @property
    def is_manipulated(self) -> bool:
        """
        Returns True if the image is considered manipulated.
        """

        return self.final_verdict in (
            FinalVerdict.MANIPULATED,
            FinalVerdict.AI_GENERATED,
            FinalVerdict.SUSPICIOUS,
        )

    @property
    def is_authentic(self) -> bool:
        """
        Returns True if the image is authentic.
        """

        return self.final_verdict == FinalVerdict.AUTHENTIC

    @property
    def confidence_percentage(self) -> float:
        """
        Confidence expressed as a percentage.
        """

        return round(
            self.confidence_assessment.confidence * 100,
            2,
        )

    @property
    def risk_percentage(self) -> float:
        """
        Risk score expressed as a percentage.
        """

        return round(
            self.risk_assessment.risk_score * 100,
            2,
        )

    @property
    def total_findings(self) -> int:
        """
        Total number of forensic findings.
        """

        return len(self.findings)

    @property
    def total_recommendations(self) -> int:
        """
        Total recommendations generated.
        """

        return len(self.recommendations)

    @property
    def has_localization(self) -> bool:
        """
        Returns True if localization results exist.
        """

        return self.localization is not None

    @property
    def has_metadata(self) -> bool:
        """
        Returns True if metadata analysis exists.
        """

        return self.metadata_summary is not None

    @property
    def has_raw_outputs(self) -> bool:
        """
        Returns True if raw model outputs were stored.
        """

        return self.raw_outputs is not None

    @property
    def inference_time_ms(self) -> Optional[float]:
        """
        Returns total inference time if available.
        """

        if self.performance_statistics is None:
            return None

        return self.performance_statistics.total_inference_time_ms
