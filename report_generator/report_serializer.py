"""
report_serializer.py

Converts a ForensicReport dataclass into the final
JSON format returned by FastAPI.

This module NEVER performs forensic analysis.
It only serializes existing pipeline outputs.
"""

from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from datetime import datetime


class ReportSerializer:
    """Serialize ForensicReport into JSON-compatible dict."""

    @staticmethod
    def _serialize(obj):
        """
        Recursively convert dataclasses, enums,
        Path objects and datetime objects into
        JSON-compatible values.
        """

        if obj is None:
            return None

        if isinstance(obj, Enum):
            return obj.value if hasattr(obj, "value") else str(obj)

        if isinstance(obj, Path):
            return str(obj)

        if isinstance(obj, datetime):
            return obj.isoformat()

        if is_dataclass(obj):
            return {
                key: ReportSerializer._serialize(value)
                for key, value in asdict(obj).items()
            }

        if isinstance(obj, list):
            return [
                ReportSerializer._serialize(item)
                for item in obj
            ]

        if isinstance(obj, dict):
            return {
                key: ReportSerializer._serialize(value)
                for key, value in obj.items()
            }

        return obj

    @staticmethod
    def to_json(report):
        """
        Convert ForensicReport to final API JSON.
        """

        return {

            "status": "success",

            "analysis": ReportSerializer._serialize(
                report.report_information
            ),

            "file_information": ReportSerializer._serialize(
                report.file_information
            ),

            "evidence_scores": ReportSerializer._serialize(
                report.evidence_scores
            ),

            "risk_assessment": ReportSerializer._serialize(
                report.risk_assessment
            ),

            "decision": ReportSerializer._serialize(
                report.decision
            ),

            "metadata": ReportSerializer._serialize(
                report.metadata_summary
            ),

            "visualization": ReportSerializer._serialize(
                report.localization
            ),

            "recommendations": ReportSerializer._serialize(
                report.recommendations
            ),

            "findings": ReportSerializer._serialize(
                report.findings
            ),

            "raw_outputs": ReportSerializer._serialize(
                report.raw_outputs
            )
        }