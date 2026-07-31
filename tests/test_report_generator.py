"""
Tests for report_generator.py
"""

from datetime import datetime

import pytest

from report_generator.report_generator import ReportGenerator
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


@pytest.fixture
def generator():
    return ReportGenerator()


@pytest.fixture
def sample_inputs():

    return dict(

        report_level=ReportLevel.DETAILED,

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
            scores={"Authentic": 0.95}
        ),

        evidence_summary=EvidenceSummary(
            spatial_score=0.91,
            frequency_score=0.92,
            context_score=0.90,
            metadata_score=0.93,
            overall_score=0.92,
        ),

        risk_assessment=RiskAssessment(
            risk_score=0.15,
            risk_level=RiskLevel.LOW,
        ),

        confidence_assessment=ConfidenceAssessment(
            confidence=0.95,
            uncertainty=0.05,
            confidence_level=ConfidenceLevel.VERY_HIGH,
        ),

        final_verdict=FinalVerdict.AUTHENTIC,
    )
def test_build_report(
    generator,
    sample_inputs,
):

    report = generator.build_report(
        **sample_inputs
    )

    assert isinstance(
        report,
        ForensicReport,
    )


def test_generate(
    generator,
    sample_inputs,
):

    report, recommendations = generator.generate(
        **sample_inputs
    )

    assert isinstance(
        report,
        ForensicReport,
    )

    assert isinstance(
        recommendations,
        list,
    )


def test_to_dict(
    generator,
    sample_inputs,
):

    report = generator.build_report(
        **sample_inputs
    )

    dictionary = generator.to_dict(
        report
    )

    assert isinstance(
        dictionary,
        dict,
    )


def test_to_json(
    generator,
    sample_inputs,
):

    report = generator.build_report(
        **sample_inputs
    )

    text = generator.to_json(
        report
    )

    assert isinstance(
        text,
        str,
    )


def test_save_json(
    generator,
    sample_inputs,
    tmp_path,
):

    report = generator.build_report(
        **sample_inputs
    )

    output = tmp_path / "report.json"

    returned = generator.save_json(
        report,
        output,
    )

    assert output.exists()

    assert returned == output


def test_pretty_print(
    generator,
    sample_inputs,
    capsys,
):

    report = generator.build_report(
        **sample_inputs
    )

    generator.pretty_print(
        report
    )

    captured = capsys.readouterr()

    assert "image.jpg" in captured.out