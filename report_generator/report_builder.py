"""
UIFAF
Universal Image Forensic Analysis Framework

Module:
Universal Forensic Report Generator (UFRG)

File:
report_builder.py

Description:
Builds immutable forensic reports from the outputs
produced by the forensic analysis pipeline.

Author:
UIFAF Project
"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from .exceptions import InvalidReportError
from .report_models import (
    ConfidenceAssessment,
    EvidenceScores,
    EvidenceSummary,
    FileInformation,
    Finding,
    ForensicReport,
    LocalizationResult,
    MetadataSummary,
    PerformanceStatistics,
    RawModelOutputs,
    ReportLevel,
    RiskAssessment,
    SystemInformation,
    ThresholdInformation,
)
from .report_validator import ReportValidator


class ReportBuilder:
    """
    Builds validated immutable forensic reports.

    The builder gathers all outputs produced by the
    forensic pipeline and assembles them into a single
    ForensicReport object.

    The builder never performs forensic analysis.
    """

    def __init__(self) -> None:
        """
        Initialize the report builder.
        """

        self._validator = ReportValidator()
    def build(
        self,
        *,
        report_level: ReportLevel,
        primary_class: str | None = None,
        secondary_class: str | None = None,
        classification_confidence: float | None = None,
        file_information: FileInformation,
        evidence_scores: EvidenceScores,
        evidence_summary: EvidenceSummary,
        risk_assessment: RiskAssessment,
        confidence_assessment: ConfidenceAssessment,
        final_verdict,
        findings: list[Finding] | None = None,
        localization: LocalizationResult | None = None,
        metadata_summary: MetadataSummary | None = None,
        raw_outputs: RawModelOutputs | None = None,
        threshold_information: ThresholdInformation | None = None,
        system_information: SystemInformation | None = None,
        performance_statistics: PerformanceStatistics | None = None,
    ) -> ForensicReport:
        """
        Build and validate a forensic report.
        """

        report = ForensicReport(
            primary_class=primary_class,
            secondary_class=secondary_class,
            classification_confidence=classification_confidence,
            report_id=self._generate_report_id(),

            timestamp=self._generate_timestamp(),

            report_level=report_level,

            file_information=file_information,

            evidence_scores=evidence_scores,

            evidence_summary=evidence_summary,

            risk_assessment=risk_assessment,

            confidence_assessment=confidence_assessment,

            final_verdict=final_verdict,

            findings=findings or [],

            localization=localization,

            metadata_summary=metadata_summary,

            raw_outputs=raw_outputs,

            threshold_information=threshold_information,

            system_information=system_information,

            performance_statistics=performance_statistics,
        )

        self._validator.validate(report)

        return report
    
    def _validate(
        self,
        report: ForensicReport,
    ) -> None:
        """
        Validate the completed forensic report.

        Parameters
        ----------
        report:
            Report to validate.

        Raises
        ------
        InvalidReportError
            If validation fails.
        """

        try:
            self._validator.validate(report)

        except Exception as exc:
            raise InvalidReportError(
                "Failed to validate forensic report."
            ) from exc
            
    def _generate_report_id(
        self,
    ) -> str:
        """
        Generate a unique report identifier.
        """

        return str(uuid4())
    
    def _generate_timestamp(
        self,
    ) -> datetime:
        """
        Generate report timestamp.
        """

        return datetime.utcnow()