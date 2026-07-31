"""
UIFAF
Universal Image Forensic Analysis Framework

Module:
Universal Forensic Report Generator

File:
report_interpreter.py

Description
-----------
Converts pipeline outputs into the immutable
ForensicReport object.

This module performs NO AI inference.

Responsibilities
----------------
1. Read PipelineContext
2. Read DecisionOutput
3. Convert outputs into report models
4. Return a ForensicReport
"""

from __future__ import annotations

from datetime import datetime
from multiprocessing import context
import platform
from uuid import uuid4

from pipeline.pipeline_models import (
    PipelineContext,
    PipelineStatistics,
)

from decision.decision_models import (
    DecisionOutput,
    FinalVerdict,
)
from preprocessing import image_info
from tests.test_report_formatter import report

from .report_models import (
    ConfidenceAssessment,
    ConfidenceLevel,
    EvidenceScores,
    EvidenceSummary,
    FileInformation,
    Finding,
    ForensicReport,
    MetadataSummary,
    PerformanceStatistics,
    RawModelOutput,
    RawModelOutputs,
    Recommendation,
    RecommendationCategory,
    RecommendationPriority,
    ReportLevel,
    RiskAssessment,
    RiskLevel,
    SystemInformation,
    ThresholdInformation,
)

from pathlib import Path
from .exceptions import ReportGenerationError

class ReportInterpreter:
    """
    Converts DecisionOutput into
    ForensicReport.
    """
    def _validate_report(
    self,
        report: ForensicReport,
    ) -> None:
   

        if report.file_information is None:
            raise ReportGenerationError(
            "Missing file information."
        )

        if report.evidence_scores is None:
            raise ReportGenerationError(
            "Missing evidence scores."
        )

        if report.confidence_assessment.confidence < 0:

            raise ReportGenerationError(
            "Invalid confidence."
        )

    def __init__(self):

        pass
    def interpret(
    self,
    context: PipelineContext,
    statistics: PipelineStatistics,
    decision: DecisionOutput,
) -> ForensicReport:
    

        try:

            file_information = self._build_file_information(
            context,
        )

            metadata_summary = self._build_metadata_summary(
            context,
        )

            evidence_scores = self._build_evidence_scores(
            decision,
        )

            evidence_summary = self._build_evidence_summary(
            context,
        )

            risk_assessment = self._build_risk(
            decision,
        )

            confidence_assessment = self._build_confidence(
            decision,
        )

            findings = self._build_findings(
            decision,
        )

            recommendations = self._build_recommendations(
            decision,
        )

            raw_outputs = self._build_raw_outputs(
            context,
        )

            threshold_information = (
            self._build_threshold_information()
        )

            system_information = (
            self._build_system_information()
        )

            performance_statistics = (
            self._build_performance_statistics(
                statistics,
            )
        )

            report = ForensicReport(

            report_id=str(uuid4()),

            timestamp=datetime.utcnow(),

            report_level=ReportLevel.DETAILED,

            file_information=file_information,

            evidence_scores=evidence_scores,

            evidence_summary=evidence_summary,

            risk_assessment=risk_assessment,

            confidence_assessment=confidence_assessment,

            final_verdict=decision.final_verdict,

            primary_class=decision.classification.primary_class,

            secondary_class=decision.classification.secondary_class,

            classification_confidence=(
                decision.classification.probability
            ),

            findings=findings,

            recommendations=recommendations,

            localization=context.localization,

            metadata_summary=metadata_summary,

            raw_outputs=raw_outputs,

            threshold_information=threshold_information,

            system_information=system_information,

            performance_statistics=performance_statistics,
        )

            self._validate_report(
            report,
)

            return report

        except Exception as exc:

            raise ReportGenerationError(
            f"Failed to generate forensic report: {exc}"
        ) from exc
    

        raise NotImplementedError
    
    def _build_file_information(
    self,
    context: PipelineContext,
) -> FileInformation:
    

        image_info = context.image_info

        return FileInformation(

        filename=image_info.filename,

        file_size_bytes=image_info.file_size,

        width=image_info.width,

        height=image_info.height,

        image_format=image_info.image_format,

        channels=image_info.channels,

        color_space=image_info.color_mode,

        resolution=f"{image_info.width} × {image_info.height}",

        extension=Path(image_info.filename).suffix.lower(),
    )


    def _build_metadata_summary(
        self,
    context: PipelineContext,
) -> MetadataSummary | None:
    

        metadata = context.metadata

        if metadata is None:
            return None

        raw_metadata = {
            str(key): str(value)
            for key, value in metadata.raw_exif.items()
    }

        return MetadataSummary(

            metadata=raw_metadata,

            exif_present=bool(metadata.raw_exif),

            metadata_modified=False,

            suspicious_fields=[],

            software=metadata.camera.software,

            camera_model=metadata.camera.model,

            gps_present=(
            metadata.gps.latitude is not None
            and metadata.gps.longitude is not None
        ),
    )


    def _build_evidence_scores(
        self,
        decision: DecisionOutput,
) -> EvidenceScores:
    

        scores = {
        evidence.image_class: evidence.probability
        for evidence in decision.evidence_scores
    }

        return EvidenceScores(
            scores=scores,
    )


    def _build_evidence_summary(
    self,
    context: PipelineContext,
) -> EvidenceSummary:
    

        reasoning = context.reasoning

        if reasoning is None:
            raise ReportGenerationError(
            "UFRE output is missing."
        )

        summary = reasoning.evidence_summary

        return EvidenceSummary(

            total_evidence=summary.total_evidence,

            average_confidence=summary.average_confidence,

            strongest_evidence=(
            summary.strongest_evidence.value
        ),

            weakest_evidence=(
            summary.weakest_evidence.value
        ),

            overall_confidence=(
            reasoning.global_confidence.confidence
        ),
    )


    def _build_risk(
    self,
    decision: DecisionOutput,
) -> RiskAssessment:
    

        risk = decision.risk_score

        if risk < 0.20:
            level = RiskLevel.MINIMAL

        elif risk < 0.40:
            level = RiskLevel.LOW

        elif risk < 0.60:
            level = RiskLevel.MODERATE

        elif risk < 0.80:
            level = RiskLevel.HIGH

        else:
            level = RiskLevel.CRITICAL

        return RiskAssessment(

        risk_score=risk,

        risk_level=level,
    )


    def _build_confidence(
    self,
    decision: DecisionOutput,
) -> ConfidenceAssessment:
    

        confidence = decision.confidence

        if confidence < 0.20:
            level = ConfidenceLevel.VERY_LOW

        elif confidence < 0.40:
            level = ConfidenceLevel.LOW

        elif confidence < 0.60:
            level = ConfidenceLevel.MEDIUM

        elif confidence < 0.80:
            level = ConfidenceLevel.HIGH

        else:
            level = ConfidenceLevel.VERY_HIGH

        return ConfidenceAssessment(

        confidence=confidence,

        uncertainty=decision.uncertainty,

        confidence_level=level,
    )


    def _build_findings(
    self,
    decision: DecisionOutput,
) -> list[Finding]:
    

        findings: list[Finding] = []

        for finding in decision.findings:

            findings.append(

            Finding(

                title=finding.title,

                description=finding.description,

                confidence=finding.confidence,

                source_module=finding.source_module,

                source_branch=finding.source_branch,

                supporting_evidence=list(
                    finding.supporting_evidence
                ),
            )

        )

        return findings


    def _build_recommendations(
    self,
    decision: DecisionOutput,
) -> list[Recommendation]:
    

        recommendations: list[Recommendation] = []

        verdict = decision.final_verdict

    # --------------------------------------------------
    # Authentic
    # --------------------------------------------------

        if verdict == FinalVerdict.AUTHENTIC:

            recommendations.append(

                Recommendation(

                message=(
                    "Image appears authentic. "
                    "No additional forensic analysis is required."
                ),

                priority=RecommendationPriority.LOW,

                category=RecommendationCategory.GENERAL,
            )

        )

    # --------------------------------------------------
    # Manipulated
    # --------------------------------------------------

        elif verdict == FinalVerdict.MANIPULATED:

            recommendations.append(

                Recommendation(

                message=(
                    "Potential manipulation detected. "
                    "Perform manual forensic verification."
                ),

                priority=RecommendationPriority.HIGH,

                category=RecommendationCategory.FORENSICS,
            )

        )

            recommendations.append(

                Recommendation(

                    message=(
                    "Avoid using this image as legal or "
                    "investigative evidence without expert review."
                ),

                    priority=RecommendationPriority.CRITICAL,

                    category=RecommendationCategory.INVESTIGATION,
            )

        )

    # --------------------------------------------------
    # AI Generated
    # --------------------------------------------------

        elif verdict == FinalVerdict.AI_GENERATED:

            recommendations.append(

            Recommendation(

                message=(
                    "Image is likely AI generated. "
                    "Verify its source before sharing."
                ),

                priority=RecommendationPriority.HIGH,

                category=RecommendationCategory.VERIFICATION,
            )

        )

    # --------------------------------------------------
    # Suspicious
    # --------------------------------------------------

        elif verdict == FinalVerdict.SUSPICIOUS:

            recommendations.append(

            Recommendation(

                message=(
                    "Evidence is inconclusive. "
                    "Perform additional forensic analysis."
                ),

                priority=RecommendationPriority.MEDIUM,

                category=RecommendationCategory.FORENSICS,
            )

        )

    # --------------------------------------------------
    # Unknown
    # --------------------------------------------------

        else:

            recommendations.append(

            Recommendation(

                message=(
                    "Unable to determine image authenticity "
                    "with sufficient confidence."
                ),

                priority=RecommendationPriority.MEDIUM,

                category=RecommendationCategory.GENERAL,
            )

        )

        return recommendations


    def _build_raw_outputs(
    self,
    context: PipelineContext,
) -> RawModelOutputs | None:
   

        reasoning = context.reasoning

        if reasoning is None:
            return None

        outputs: list[RawModelOutput] = []

        for stage in reasoning.stages:

            outputs.append(

                RawModelOutput(

                module="UFRE",

                branch=stage.stage,

                prediction="Forensic Evidence",

                probability=stage.confidence.confidence,

                confidence=stage.confidence.confidence,

                threshold=0.50,
            )

        )

        return RawModelOutputs(
        outputs=outputs,
    )


    def _build_threshold_information(
        self,
) -> ThresholdInformation:
    

        return ThresholdInformation(

            thresholds={

            "High Confidence":0.85,

            "Medium Confidence":0.60,

            "Low Confidence":0.35,

            "High Risk":0.75,

            "Medium Risk":0.45,

        }

    )


    def _build_system_information(
    self,
) -> SystemInformation:
    

        return SystemInformation(

        framework_name="UIFAF",

        framework_version="1.0",

        backbone="ConvNeXt V2",

        attention_module="EMA Attention",

        reasoning_engine="UFRE",

        python_version=platform.python_version(),

        device="CPU",
    )


    def _build_performance_statistics(
    self,
    statistics: PipelineStatistics,
) -> PerformanceStatistics:
    

        execution = statistics.execution_times

        preprocessing = execution.get(
        "preprocessing",
        0.0,
    )

        feature = (

        execution.get("backbone",0.0)

        + execution.get("attention",0.0)

        + execution.get("representation",0.0)

    )

        reasoning = execution.get(
        "ufre",
        0.0,
    )

        total = sum(
        execution.values()
    )

        return PerformanceStatistics(

        preprocessing_time_ms=preprocessing,

        feature_extraction_time_ms=feature,

        reasoning_time_ms=reasoning,

        total_inference_time_ms=total,

        memory_usage_mb=None,
    )