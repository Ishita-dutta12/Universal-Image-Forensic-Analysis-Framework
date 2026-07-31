"""
UIFAF
Universal Image Forensic Analysis Framework

Module:
Universal Forensic Report Generator (UFRG)

File:
report_generator.py

Description:
Facade for generating complete forensic reports.

Author:
UIFAF Project
"""
from __future__ import annotations
from .report_builder import ReportBuilder


#from tests.test_report_formatter import report


from .recommendation_engine import RecommendationEngine
from .report_formatter import ReportFormatter
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


class ReportGenerator:
    """
    Facade for the Universal Forensic Report Generator.

    This class coordinates the report building,
    recommendation generation and report formatting
    into a single interface.
    """

    def __init__(self) -> None:
        """
        Initialize report generation components.
        """

        #self._builder = ReportBuilder()
        self._engine = RecommendationEngine()
        self._formatter = ReportFormatter()

    

    def generate_recommendations(
        self,
        report: ForensicReport,
    ):
        """
        Generate recommendations for a report.
        """

        return self._engine.generate(report)

    def to_dict(
        self,
        report: ForensicReport,
    ):
        """
        Convert report to dictionary.
        """

        return self._formatter.to_dict(report)

    def to_json(
        self,
        report: ForensicReport,
        *,
        indent: int = 4,
    ) -> str:
        """
        Convert report to JSON.
        """

        return self._formatter.to_json(
            report,
            indent=indent,
        )

    def save_json(
        self,
        report: ForensicReport,
        output_path,
    ):
        """
        Save report as JSON.
        """

        return self._formatter.save_json(
            report,
            output_path,
        )

    def pretty_print(
        self,
        report: ForensicReport,
    ) -> None:
        """
        Print report.
        """

        self._formatter.pretty_print(report)

    def finalize(
    self,
    report: ForensicReport,
) -> tuple[
    ForensicReport,
    list,
]:
    

        recommendations = (
        self.generate_recommendations(
            report,
        )
    )

        return (
        report,
        recommendations,
    )