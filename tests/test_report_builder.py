"""
Tests for report_builder.py
"""

from datetime import datetime

import pytest

from report_generator.exceptions import InvalidReportError
from report_generator.report_builder import ReportBuilder
from report_generator.report_models import (
    ConfidenceAssessment,
    ConfidenceLevel,
    EvidenceScores,
    EvidenceSummary,
    FileInformation,
    FinalVerdict,
    Finding,
    ForensicReport,
    LocalizationResult,
    MetadataSummary,
    PerformanceStatistics,
    RawModelOutput,
    RawModelOutputs,
    ReportLevel,
    RiskAssessment,
    RiskLevel,
    SystemInformation,
    ThresholdInformation,
)


@pytest.fixture
def builder():
    return ReportBuilder()


@pytest.fixture
def file_information():
    return FileInformation(
        filename="image.jpg",
        file_size_bytes=1024,
        width=512,
        height=512,
        image_format="JPEG",
        channels=3,
        color_space="RGB",
        resolution="512x512",
        extension=".jpg",
    )


@pytest.fixture
def evidence_scores():
    return EvidenceScores(
        scores={
            "authentic": 0.92,
            "ai": 0.08,
        }
    )


@pytest.fixture
def evidence_summary():
    return EvidenceSummary(
        spatial_score=0.91,
        frequency_score=0.89,
        context_score=0.93,
        metadata_score=0.95,
        overall_score=0.92,
    )


@pytest.fixture
def risk():
    return RiskAssessment(
        risk_score=0.18,
        risk_level=RiskLevel.LOW,
    )


@pytest.fixture
def confidence():
    return ConfidenceAssessment(
        confidence=0.94,
        uncertainty=0.06,
        confidence_level=ConfidenceLevel.VERY_HIGH,
    )
def test_build_report(
    builder,
    file_information,
    evidence_scores,
    evidence_summary,
    risk,
    confidence,
):
    report = builder.build(
        report_level=ReportLevel.DETAILED,
        file_information=file_information,
        evidence_scores=evidence_scores,
        evidence_summary=evidence_summary,
        risk_assessment=risk,
        confidence_assessment=confidence,
        final_verdict=FinalVerdict.AUTHENTIC,
    )

    assert isinstance(report, ForensicReport)

    assert report.final_verdict == FinalVerdict.AUTHENTIC

    assert report.report_level == ReportLevel.DETAILED
def test_report_id_generated(
    builder,
    file_information,
    evidence_scores,
    evidence_summary,
    risk,
    confidence,
):
    report = builder.build(
        report_level=ReportLevel.BASIC,
        file_information=file_information,
        evidence_scores=evidence_scores,
        evidence_summary=evidence_summary,
        risk_assessment=risk,
        confidence_assessment=confidence,
        final_verdict=FinalVerdict.AUTHENTIC,
    )

    assert report.report_id

    assert isinstance(report.report_id, str)


def test_timestamp_generated(
    builder,
    file_information,
    evidence_scores,
    evidence_summary,
    risk,
    confidence,
):
    report = builder.build(
        report_level=ReportLevel.BASIC,
        file_information=file_information,
        evidence_scores=evidence_scores,
        evidence_summary=evidence_summary,
        risk_assessment=risk,
        confidence_assessment=confidence,
        final_verdict=FinalVerdict.AUTHENTIC,
    )

    assert isinstance(
        report.timestamp,
        datetime,
    )
    
def test_optional_fields(
    builder,
    file_information,
    evidence_scores,
    evidence_summary,
    risk,
    confidence,
):
    metadata = MetadataSummary(
        metadata={},
        exif_present=True,
        metadata_modified=False,
        suspicious_fields=[],
    )

    localization = LocalizationResult(
        detected_regions=2,
        manipulated_area_percentage=25,
    )

    report = builder.build(
        report_level=ReportLevel.RESEARCH,
        file_information=file_information,
        evidence_scores=evidence_scores,
        evidence_summary=evidence_summary,
        risk_assessment=risk,
        confidence_assessment=confidence,
        final_verdict=FinalVerdict.AUTHENTIC,
        metadata_summary=metadata,
        localization=localization,
    )

    assert report.metadata_summary is metadata

    assert report.localization is localization
    
def test_raw_outputs(
    builder,
    file_information,
    evidence_scores,
    evidence_summary,
    risk,
    confidence,
):
    raw = RawModelOutputs(
        outputs=[
            RawModelOutput(
                module="CNN",
                branch="Spatial",
                prediction="Authentic",
                probability=0.94,
                confidence=0.91,
                threshold=0.50,
            )
        ]
    )

    report = builder.build(
        report_level=ReportLevel.RESEARCH,
        file_information=file_information,
        evidence_scores=evidence_scores,
        evidence_summary=evidence_summary,
        risk_assessment=risk,
        confidence_assessment=confidence,
        final_verdict=FinalVerdict.AUTHENTIC,
        raw_outputs=raw,
    )

    assert report.raw_outputs is raw
    
def test_findings(
    builder,
    file_information,
    evidence_scores,
    evidence_summary,
    risk,
    confidence,
):
    findings = [
        Finding(
            title="Forgery",
            description="Possible manipulation",
            confidence=0.95,
            source_module="UFRE",
            source_branch="Fusion",
        )
    ]

    report = builder.build(
        report_level=ReportLevel.DETAILED,
        file_information=file_information,
        evidence_scores=evidence_scores,
        evidence_summary=evidence_summary,
        risk_assessment=risk,
        confidence_assessment=confidence,
        final_verdict=FinalVerdict.MANIPULATED,
        findings=findings,
    )

    assert len(report.findings) == 1
def test_invalid_report(builder):

    with pytest.raises(Exception):
        builder.build(
            report_level=ReportLevel.DETAILED,
            file_information=None,
            evidence_scores=None,
            evidence_summary=None,
            risk_assessment=None,
            confidence_assessment=None,
            final_verdict=None,
        )