"""
UIFAF
Universal Image Forensic Analysis Framework

Module:
Universal Forensic Report Generator (UFRG)

File:
recommendation_engine.py

Description:
Generates intelligent forensic recommendations based on
the validated forensic report.

Author:
UIFAF Project
"""

from __future__ import annotations

from typing import List

from .exceptions import InvalidReportError
from .report_models import (
    ConfidenceAssessment,
    FinalVerdict,
    Finding,
    ForensicReport,
    Recommendation,
    RecommendationCategory,
    RecommendationPriority,
    RiskAssessment,
    RiskLevel,
    LocalizationResult,
    MetadataSummary,
    
)


class RecommendationEngine:
    """
    Generates actionable recommendations from a forensic report.

    The recommendation engine does not perform any forensic
    analysis. It only interprets the final report and produces
    human-readable recommendations.
    """

    
    def _from_verdict(
        self,
        report: ForensicReport,
    ) -> List[Recommendation]:
        """
        Generate recommendations from the final verdict.
        """

        verdict = report.final_verdict

        recommendations: List[Recommendation] = []

        if verdict == FinalVerdict.AUTHENTIC:

            recommendations.append(
                Recommendation(
                    message="No additional forensic investigation is required.",
                    priority=RecommendationPriority.LOW,
                    category=RecommendationCategory.GENERAL,
                )
            )

        elif verdict == FinalVerdict.AI_GENERATED:

            recommendations.extend(
                [
                    Recommendation(
                        message="Verify the source before publication.",
                        priority=RecommendationPriority.HIGH,
                        category=RecommendationCategory.VERIFICATION,
                    ),
                    Recommendation(
                        message="Disclose AI-generated origin where appropriate.",
                        priority=RecommendationPriority.HIGH,
                        category=RecommendationCategory.VERIFICATION,
                    ),
                ]
            )

        elif verdict == FinalVerdict.MANIPULATED:

            recommendations.extend(
                [
                    Recommendation(
                        message="Conduct manual forensic review.",
                        priority=RecommendationPriority.CRITICAL,
                        category=RecommendationCategory.FORENSICS,
                    ),
                    Recommendation(
                        message="Compare with the original image whenever possible.",
                        priority=RecommendationPriority.CRITICAL,
                        category=RecommendationCategory.FORENSICS,
                    ),
                    Recommendation(
                        message="Do not rely on this image as evidence until verified.",
                        priority=RecommendationPriority.CRITICAL,
                        category=RecommendationCategory.FORENSICS,
                    ),
                ]
            )

        elif verdict == FinalVerdict.SUSPICIOUS:

            recommendations.extend(
                [
                    Recommendation(
                        message="Run additional forensic analysis.",
                        priority=RecommendationPriority.HIGH,
                        category=RecommendationCategory.INVESTIGATION,
                    ),
                    Recommendation(
                        message="Acquire a higher-quality source image.",
                        priority=RecommendationPriority.HIGH,
                        category=RecommendationCategory.INVESTIGATION,
                    ),
                ]
            )

        elif verdict == FinalVerdict.UNKNOWN:

            recommendations.append(
                Recommendation(
                    message="Unable to determine image authenticity. Additional investigation is recommended.",
                    priority=RecommendationPriority.MEDIUM,
                    category=RecommendationCategory.INVESTIGATION,
                )
            )

        return recommendations
    def _from_risk(
        self,
        risk: RiskAssessment,
    ) -> List[Recommendation]:
        """
        Generate recommendations from the overall
        manipulation risk assessment.
        """

        recommendations: List[Recommendation] = []

        if risk.risk_level == RiskLevel.HIGH:

            recommendations.append(
                Recommendation(
                    message="Manual forensic investigation is recommended.",
                    priority=RecommendationPriority.HIGH,
                    category=RecommendationCategory.FORENSICS,
                )
            )

        elif risk.risk_level == RiskLevel.CRITICAL:

            recommendations.extend(
                [
                    Recommendation(
                        message="Treat this image as highly suspicious.",
                        priority=RecommendationPriority.CRITICAL,
                        category=RecommendationCategory.FORENSICS,
                    ),
                    Recommendation(
                        message="Escalate the image for expert forensic review.",
                        priority=RecommendationPriority.CRITICAL,
                        category=RecommendationCategory.INVESTIGATION,
                    ),
                ]
            )

        return recommendations

    def _from_confidence(
        self,
        confidence: ConfidenceAssessment,
    ) -> List[Recommendation]:
        """
        Generate recommendations based on model confidence.
        """

        recommendations: List[Recommendation] = []

        if confidence.confidence < 0.60:

            recommendations.append(
                Recommendation(
                    message="Run additional forensic models to improve confidence.",
                    priority=RecommendationPriority.MEDIUM,
                    category=RecommendationCategory.VERIFICATION,
                )
            )

        if confidence.uncertainty > 0.40:

            recommendations.append(
                Recommendation(
                    message="High uncertainty detected. Manual verification is recommended.",
                    priority=RecommendationPriority.HIGH,
                    category=RecommendationCategory.VERIFICATION,
                )
            )

        return recommendations

    def _from_metadata(
        self,
        metadata: MetadataSummary | None,
    ) -> List[Recommendation]:
        """
        Generate recommendations from metadata analysis.
        """

        recommendations: List[Recommendation] = []

        if metadata is None:
            return recommendations

        if metadata.metadata_modified:

            recommendations.append(
                Recommendation(
                    message="Inspect metadata inconsistencies before trusting the image.",
                    priority=RecommendationPriority.HIGH,
                    category=RecommendationCategory.VERIFICATION,
                )
            )

        if metadata.suspicious_fields:

            recommendations.append(
                Recommendation(
                    message="Verify suspicious EXIF fields manually.",
                    priority=RecommendationPriority.MEDIUM,
                    category=RecommendationCategory.VERIFICATION,
                )
            )

        if (
            not metadata.exif_present
            and metadata.camera_model is None
        ):

            recommendations.append(
                Recommendation(
                    message="Original camera metadata is unavailable.",
                    priority=RecommendationPriority.LOW,
                    category=RecommendationCategory.GENERAL,
                )
            )

        return recommendations
    def _from_localization(
        self,
        localization: LocalizationResult | None,
    ) -> List[Recommendation]:
        """
        Generate recommendations from localization results.
        """

        recommendations: List[Recommendation] = []

        if localization is None:
            return recommendations

        if localization.detected_regions > 0:

            recommendations.append(
                Recommendation(
                    message=(
                        "Inspect the highlighted manipulated "
                        "regions carefully."
                    ),
                    priority=RecommendationPriority.HIGH,
                    category=RecommendationCategory.FORENSICS,
                )
            )

        if (
            localization.manipulated_area_percentage is not None
            and localization.manipulated_area_percentage > 50
        ):

            recommendations.append(
                Recommendation(
                    message=(
                        "Large manipulated area detected. "
                        "Perform a comprehensive forensic review."
                    ),
                    priority=RecommendationPriority.CRITICAL,
                    category=RecommendationCategory.FORENSICS,
                )
            )

        return recommendations

    def _from_findings(
        self,
        findings: List[Finding],
    ) -> List[Recommendation]:
        """
        Generate recommendations from forensic findings.
        """

        recommendations: List[Recommendation] = []

        for finding in findings:

            if finding.confidence >= 0.90:

                recommendations.append(
                    Recommendation(
                        message=f"Review finding: {finding.title}",
                        priority=RecommendationPriority.MEDIUM,
                        category=RecommendationCategory.INVESTIGATION,
                    )
                )

        return recommendations

    def _deduplicate(
        self,
        recommendations: List[Recommendation],
    ) -> List[Recommendation]:
        """
        Remove duplicate recommendations.
        """

        unique: dict[
        tuple[str, RecommendationPriority, RecommendationCategory],
        Recommendation,
        ] = {}

        for recommendation in recommendations:

            key = (
                recommendation.message,
                recommendation.priority,
                recommendation.category,
            )

            if key not in unique:
                unique[key] = recommendation

        return list(unique.values())

    def _sort(
        self,
        recommendations: List[Recommendation],
    ) -> List[Recommendation]:
        """
        Sort recommendations by priority.

        Order:
        CRITICAL
        HIGH
        MEDIUM
        LOW
        """

        priority_order = {
            RecommendationPriority.CRITICAL: 0,
            RecommendationPriority.HIGH: 1,
            RecommendationPriority.MEDIUM: 2,
            RecommendationPriority.LOW: 3,
        }

        return sorted(
            recommendations,
            key=lambda recommendation: priority_order[
                recommendation.priority
            ],
        )