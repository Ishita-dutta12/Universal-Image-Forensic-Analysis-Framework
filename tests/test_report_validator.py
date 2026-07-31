"""
Tests for report_validator.py
"""

from datetime import datetime
from dataclasses import replace
import pytest

from report_generator.exceptions import InvalidReportError
from report_generator.report_models import (
    ConfidenceAssessment,
    ConfidenceLevel,
    EvidenceScores,
    EvidenceSummary,
    FileInformation,
    FinalVerdict,
    ForensicReport,
    ReportLevel,
    RiskAssessment,
    RiskLevel,
)
from report_generator.report_validator import ReportValidator


@pytest.fixture
def valid_report():
    """
    Returns a minimal valid report.
    """

    return ForensicReport(
        report_id="RPT-001",

        timestamp=datetime.now(),

        report_level=ReportLevel.BASIC,

        file_information=FileInformation(
            filename="image.jpg",
            file_size_bytes=1024,
            width=512,
            height=512,
            image_format="JPEG",
            channels=3,
            color_space="RGB",
            resolution="512x512",
            extension=".jpg",
        ),

        evidence_scores=EvidenceScores(
            scores={
                "Authentic": 0.95,
            }
        ),

        evidence_summary=EvidenceSummary(
            spatial_score=0.9,
            frequency_score=0.8,
            context_score=0.85,
            metadata_score=0.95,
            overall_score=0.9,
        ),

        risk_assessment=RiskAssessment(
            risk_score=0.1,
            risk_level=RiskLevel.LOW,
        ),

        confidence_assessment=ConfidenceAssessment(
            confidence=0.97,
            uncertainty=0.03,
            confidence_level=ConfidenceLevel.VERY_HIGH,
        ),

        final_verdict=FinalVerdict.AUTHENTIC,
    )
    
def test_validate_valid_report(valid_report):
    """
    Validator should accept a valid report.
    """

    validator = ReportValidator()

    validator.validate(valid_report)
def test_invalid_filename(valid_report):
    """
    Validator should reject an empty filename.
    """

    bad_info = replace(
        valid_report.file_information,
        filename=""
    )

    bad_report = replace(
        valid_report,
        file_information=bad_info
    )

    validator = ReportValidator()

    with pytest.raises(InvalidReportError):
        validator.validate(bad_report)
def test_invalid_risk_score(valid_report):
    """
    Validator should reject an invalid risk score.
    """

    bad_risk = replace(
        valid_report.risk_assessment,
        risk_score=1.5
    )

    bad_report = replace(
        valid_report,
        risk_assessment=bad_risk
    )

    validator = ReportValidator()

    with pytest.raises(InvalidReportError):
        validator.validate(bad_report)
        
def test_invalid_confidence(valid_report):
    """
    Validator should reject an invalid confidence value.
    """

    bad_confidence = replace(
        valid_report.confidence_assessment,
        confidence=-0.1
    )

    bad_report = replace(
        valid_report,
        confidence_assessment=bad_confidence
    )

    validator = ReportValidator()

    with pytest.raises(InvalidReportError):
        validator.validate(bad_report)
        
def test_invalid_score(valid_report):
    """
    Validator should reject evidence scores outside [0, 1].
    """

    bad_scores = replace(
        valid_report.evidence_scores,
        scores={
            "Authentic": 1.2
        }
    )

    bad_report = replace(
        valid_report,
        evidence_scores=bad_scores
    )

    validator = ReportValidator()

    with pytest.raises(InvalidReportError):
        validator.validate(bad_report)