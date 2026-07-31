"""
Tests for report_formatter.py
"""

import json

import pytest

from report_generator.exceptions import InvalidReportError
from report_generator.report_formatter import ReportFormatter
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
from datetime import datetime


@pytest.fixture
def formatter():
    return ReportFormatter()


@pytest.fixture
def report():

    return ForensicReport(

        report_id="REPORT001",

        timestamp=datetime.now(),

        report_level=ReportLevel.DETAILED,

        file_information=FileInformation(
            filename="image.jpg",
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
            scores={
                "Authentic": 0.95
            }
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
    
def test_to_dict(
    formatter,
    report,
):

    data = formatter.to_dict(report)

    assert isinstance(data, dict)

    assert data["report_id"] == "REPORT001"

    assert data["final_verdict"] == FinalVerdict.AUTHENTIC
def test_to_json(
    formatter,
    report,
):

    json_string = formatter.to_json(report)

    assert isinstance(
        json_string,
        str,
    )

    data = json.loads(json_string)

    assert data["report_id"] == "REPORT001"
def test_save_json(
    formatter,
    report,
    tmp_path,
):

    output = tmp_path / "report.json"

    returned = formatter.save_json(
        report,
        output,
    )

    assert output.exists()

    assert returned == output

    text = output.read_text()

    assert "REPORT001" in text
    
def test_pretty_print(
    formatter,
    report,
    capsys,
):

    formatter.pretty_print(report)

    captured = capsys.readouterr()

    assert "REPORT001" in captured.out
    
def test_none_report_dict(
    formatter,
):

    with pytest.raises(
        InvalidReportError
    ):
        formatter.to_dict(None)


def test_none_report_json(
    formatter,
):

    with pytest.raises(
        InvalidReportError
    ):
        formatter.to_json(None)


def test_none_report_print(
    formatter,
):

    with pytest.raises(
        InvalidReportError
    ):
        formatter.pretty_print(None)


def test_none_report_save(
    formatter,
    tmp_path,
):

    with pytest.raises(
        InvalidReportError
    ):
        formatter.save_json(
            None,
            tmp_path / "report.json",
        )