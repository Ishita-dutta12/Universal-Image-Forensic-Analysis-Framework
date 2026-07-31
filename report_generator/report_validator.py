"""
UIFAF
Universal Image Forensic Analysis Framework

Module:
Universal Forensic Report Generator (UFRG)

File:
report_validator.py

Description:
Validates a ForensicReport before it is formatted,
exported, or returned by the framework.

This module performs structural validation only.
It does not modify report data.
"""

from __future__ import annotations
from logging import info

from tensorboard import summary

from tensorboard import summary

from .exceptions import InvalidReportError
import metadata
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
    Recommendation,
    RiskAssessment,
    ThresholdInformation,
)


class ReportValidator:
    """
    Validates a complete forensic report.

    Public API
    ----------
    validate(report)

    Raises
    ------
    InvalidReportError
        If any section of the report is invalid.
    """

    def validate(self, report: ForensicReport) -> None:
        """
        Validate every section of the report.

        Parameters
        ----------
        report:
            Forensic report to validate.

        Raises
        ------
        InvalidReportError
            If validation fails.
        """

        self._validate_file_information(
            report.file_information
        )

        self._validate_scores(
            report.evidence_scores
        )

        self._validate_evidence_summary(
            report.evidence_summary
        )

        self._validate_risk(
            report.risk_assessment
        )

        self._validate_confidence(
            report.confidence_assessment
        )

        self._validate_findings(
            report.findings
        )

        self._validate_recommendations(
            report.recommendations
        )

        if report.localization is not None:
            self._validate_localization(
                report.localization
            )

        if report.metadata_summary is not None:
            self._validate_metadata(
                report.metadata_summary
            )

        if report.raw_outputs is not None:
            self._validate_raw_outputs(
                report.raw_outputs
            )

        if report.threshold_information is not None:
            self._validate_thresholds(
                report.threshold_information
            )

        if report.performance_statistics is not None:
            self._validate_performance(
                report.performance_statistics
            )
    def _validate_file_information(
        self,
        info: FileInformation,
    ) -> None:
        """
        Validate file information.
        """

        if not info.filename:
            raise InvalidReportError(
                "Filename cannot be empty."
            )

        if info.file_size_bytes <= 0:
            raise InvalidReportError(
                "File size must be greater than zero."
            )

        if info.width <= 0:
            raise InvalidReportError(
                "Image width must be greater than zero."
            )

        if info.height <= 0:
            raise InvalidReportError(
                "Image height must be greater than zero."
            )

        if info.channels <= 0:
            raise InvalidReportError(
                "Channels must be greater than zero."
            )

        if not info.image_format:
            raise InvalidReportError(
                "Image format cannot be empty."
            )

        if not info.extension:
            raise InvalidReportError(
                "Extension cannot be empty."
            )
    def _validate_scores(
        self,
        scores: EvidenceScores,
    ) -> None:
        """
        Validate evidence scores.
        """

        for label, value in scores.scores.items():

            if value < 0.0 or value > 1.0:

                raise InvalidReportError(
                    f"Evidence score '{label}' "
                    f"must be between 0 and 1."
                )
    def _validate_evidence_summary(
        self,
        summary: EvidenceSummary,
    ) -> None:
        """
        Validate evidence summary.
        """

        values = [
            summary.spatial_score,
            summary.frequency_score,
            summary.context_score,
            summary.metadata_score,
            summary.overall_score,
        ]

        for value in values:

            if value < 0.0 or value > 1.0:

                raise InvalidReportError(
                    "Evidence summary scores "
                    "must be between 0 and 1."
                )
    def _validate_risk(
        self,
        risk: RiskAssessment,
    ) -> None:
        """
        Validate risk assessment.
        """

        if risk.risk_score < 0.0 or risk.risk_score > 1.0:

            raise InvalidReportError(
                "Risk score must be between 0 and 1."
            )
    def _validate_confidence(
        self,
        confidence: ConfidenceAssessment,
    ) -> None:
        """
        Validate confidence assessment.
        """

        if (
            confidence.confidence < 0.0
            or confidence.confidence > 1.0
        ):
            raise InvalidReportError(
                "Confidence must be between 0 and 1."
            )

        if (
            confidence.uncertainty < 0.0
            or confidence.uncertainty > 1.0
        ):
            raise InvalidReportError(
                "Uncertainty must be between 0 and 1."
            )
    def _validate_findings(
        self,
        findings: list[Finding],
    ) -> None:
        """
        Validate forensic findings.
        """

        for finding in findings:

            if not finding.title:
                raise InvalidReportError(
                    "Finding title cannot be empty."
                )

            if not finding.description:
                raise InvalidReportError(
                    "Finding description cannot be empty."
                )

            if (
                finding.confidence < 0.0
                or finding.confidence > 1.0
            ):
                raise InvalidReportError(
                    "Finding confidence must be "
                    "between 0 and 1."
                )

            if not finding.source_module:
                raise InvalidReportError(
                    "Finding source module "
                    "cannot be empty."
                )

            if not finding.source_branch:
                raise InvalidReportError(
                    "Finding source branch "
                    "cannot be empty."
                )
    def _validate_recommendations(
        self,
        recommendations: list[Recommendation],
    ) -> None:
        """
        Validate recommendations.
        """

        for recommendation in recommendations:

            if not recommendation.message:
                raise InvalidReportError(
                    "Recommendation message "
                    "cannot be empty."
                )
    def _validate_localization(
        self,
        localization: LocalizationResult,
    ) -> None:
        """
        Validate localization results.
        """

        if (
            localization.manipulated_area_percentage
            is not None
        ):
            if (
                localization.manipulated_area_percentage < 0.0
                or localization.manipulated_area_percentage > 100.0
            ):
                raise InvalidReportError(
                    "Manipulated area percentage "
                    "must be between 0 and 100."
                )

        if localization.detected_regions < 0:
            raise InvalidReportError(
                "Detected regions cannot be negative."
            )

        if localization.peak_confidence is not None:

            if (
                localization.peak_confidence < 0.0
                or localization.peak_confidence > 1.0
            ):
                raise InvalidReportError(
                    "Peak confidence must "
                    "be between 0 and 1."
                )
    def _validate_metadata(
        self,
        metadata: MetadataSummary,
    ) -> None:
        """
        Validate metadata summary.
        """

        for key, value in metadata.metadata.items():

            if not isinstance(key, str):
                raise InvalidReportError(
                    "Metadata key must be string."
                )

            if not isinstance(value, str):
                raise InvalidReportError(
                    "Metadata value must be string."
                )
    def _validate_raw_outputs(
        self,
        outputs: RawModelOutputs,
    ) -> None:
        """
        Validate raw model outputs.
        """

        for output in outputs.outputs:

            if (
                output.probability < 0.0
                or output.probability > 1.0
            ):
                raise InvalidReportError(
                    "Probability must be "
                    "between 0 and 1."
                )

            if (
                output.confidence < 0.0
                or output.confidence > 1.0
            ):
                raise InvalidReportError(
                    "Confidence must be "
                    "between 0 and 1."
                )

            if (
                output.threshold < 0.0
                or output.threshold > 1.0
            ):
                raise InvalidReportError(
                    "Threshold must be "
                    "between 0 and 1."
                )
    def _validate_thresholds(
        self,
        thresholds: ThresholdInformation,
    ) -> None:
        """
        Validate threshold values.
        """

        for name, value in thresholds.thresholds.items():

            if value < 0.0 or value > 1.0:

                raise InvalidReportError(
                    f"Threshold '{name}' "
                    f"must be between 0 and 1."
                )
    def _validate_performance(
        self,
        performance: PerformanceStatistics,
    ) -> None:
        """
        Validate performance statistics.
        """

        if performance.preprocessing_time_ms < 0:
            raise InvalidReportError(
                "Preprocessing time "
                "cannot be negative."
            )

        if performance.feature_extraction_time_ms < 0:
            raise InvalidReportError(
                "Feature extraction time "
                "cannot be negative."
            )

        if performance.reasoning_time_ms < 0:
            raise InvalidReportError(
                "Reasoning time "
                "cannot be negative."
            )

        if performance.total_inference_time_ms < 0:
            raise InvalidReportError(
                "Total inference time "
                "cannot be negative."
            )

        if (
            performance.memory_usage_mb is not None
            and performance.memory_usage_mb < 0
        ):
            raise InvalidReportError(
                "Memory usage cannot be negative."
            )
    