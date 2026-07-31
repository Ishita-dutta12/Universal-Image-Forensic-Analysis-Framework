"""
Tests for recommendation_engine.py
"""

# Ensure project root is on sys.path so "reporting" package can be resolved when tests
# are run from the tests directory or by linters that don't set PYTHONPATH.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import datetime

import pytest
from dataclasses import replace
from report_generator.exceptions import InvalidReportError
from report_generator.recommendation_engine import RecommendationEngine
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
    RecommendationCategory,
    RecommendationPriority,
    ReportLevel,
    RiskAssessment,
    RiskLevel,
)


@pytest.fixture
def engine():
    return RecommendationEngine()


@pytest.fixture
def base_report():

    return ForensicReport(

        report_id="REPORT001",

        timestamp=datetime.now(),

        report_level=ReportLevel.DETAILED,

        file_information=FileInformation(
            filename="img.jpg",
            file_size_bytes=100,
            width=512,
            height=512,
            image_format="JPEG",
            channels=3,
            color_space="RGB",
            resolution="512x512",
            extension=".jpg",
        ),

        evidence_scores=EvidenceScores(
            scores={"Authentic": 0.95}
        ),

        evidence_summary=EvidenceSummary(
            spatial_score=0.9,
            frequency_score=0.9,
            context_score=0.9,
            metadata_score=0.9,
            overall_score=0.9,
        ),

        risk_assessment=RiskAssessment(
            risk_score=0.2,
            risk_level=RiskLevel.LOW,
        ),

        confidence_assessment=ConfidenceAssessment(
            confidence=0.95,
            uncertainty=0.05,
            confidence_level=ConfidenceLevel.VERY_HIGH,
        ),

        final_verdict=FinalVerdict.AUTHENTIC,
    )
def test_generate_none_report(engine):

    with pytest.raises(InvalidReportError):
        engine.generate(None)


def test_authentic_report(engine, base_report):

    recommendations = engine.generate(base_report)

    assert len(recommendations) >= 1

    assert recommendations[-1].category == RecommendationCategory.GENERAL


def test_ai_generated(engine, base_report):

    report = replace(
        base_report,
        final_verdict=FinalVerdict.AI_GENERATED,
    )

    recommendations = engine.generate(report)

    assert any(
        r.priority == RecommendationPriority.HIGH
        for r in recommendations
    )


def test_manipulated(engine, base_report):

    report = replace(
        base_report,
        final_verdict=FinalVerdict.MANIPULATED,
    )

    recommendations = engine.generate(report)

    assert any(
        r.priority == RecommendationPriority.CRITICAL
        for r in recommendations
    )


def test_unknown(engine, base_report):

    report = base_report.__class__(
        **{
            **base_report.__dict__,
            "final_verdict": FinalVerdict.UNKNOWN,
        }
    )

    recommendations = engine.generate(report)

    assert any(
        r.priority == RecommendationPriority.MEDIUM
        for r in recommendations
    )
def test_high_risk(engine):

    recommendations = engine._from_risk(
        RiskAssessment(
            risk_score=0.8,
            risk_level=RiskLevel.HIGH,
        )
    )

    assert len(recommendations) == 1


def test_critical_risk(engine):

    recommendations = engine._from_risk(
        RiskAssessment(
            risk_score=0.99,
            risk_level=RiskLevel.CRITICAL,
        )
    )

    assert len(recommendations) == 2


def test_low_confidence(engine):

    confidence = ConfidenceAssessment(
        confidence=0.45,
        uncertainty=0.10,
        confidence_level=ConfidenceLevel.LOW,
    )

    recommendations = engine._from_confidence(confidence)

    assert len(recommendations) == 1


def test_high_uncertainty(engine):

    confidence = ConfidenceAssessment(
        confidence=0.80,
        uncertainty=0.60,
        confidence_level=ConfidenceLevel.HIGH,
    )

    recommendations = engine._from_confidence(confidence)

    assert len(recommendations) == 1
    
def test_metadata_modified(engine):

    metadata = MetadataSummary(
        metadata={},
        exif_present=True,
        metadata_modified=True,
        suspicious_fields=[],
    )

    recommendations = engine._from_metadata(metadata)

    assert len(recommendations) == 1


def test_metadata_suspicious_fields(engine):

    metadata = MetadataSummary(
        metadata={},
        exif_present=True,
        metadata_modified=False,
        suspicious_fields=["Software"],
    )

    recommendations = engine._from_metadata(metadata)

    assert len(recommendations) == 1


def test_missing_metadata(engine):

    recommendations = engine._from_metadata(None)

    assert recommendations == []
    
def test_localization(engine):

    localization = LocalizationResult(
        detected_regions=2,
        manipulated_area_percentage=65,
    )

    recommendations = engine._from_localization(localization)

    assert len(recommendations) == 2


def test_findings(engine):

    findings = [
        Finding(
            title="Forgery",
            description="Detected",
            confidence=0.95,
            source_module="UFRE",
            source_branch="Fusion",
        )
    ]

    recommendations = engine._from_findings(findings)

    assert len(recommendations) == 1


def test_remove_duplicates(engine, base_report):

    rec = engine._from_verdict(base_report)

    duplicated = rec + rec

    unique = engine._deduplicate(duplicated)

    assert len(unique) == len(rec)

def test_sort(engine, base_report):

    recommendations = engine._from_verdict(base_report)

    recommendations = engine._sort(recommendations)

    assert isinstance(recommendations, list)