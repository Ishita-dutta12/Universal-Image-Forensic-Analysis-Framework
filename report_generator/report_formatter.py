"""
UIFAF
Universal Image Forensic Analysis Framework

Module:
Universal Forensic Report Generator (UFRG)

File:
report_formatter.py

Description:
Formats forensic reports into
dictionary, JSON and console output.

Author:
UIFAF Project
"""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from dataclasses import is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

#from tests.test_report_formatter import report

from .exceptions import InvalidReportError
from .report_models import ForensicReport


class ReportFormatter:
    """
    Formats forensic reports into multiple
    output representations.
    """
    def _serialize(
    self,
    obj: Any,
) -> Any:

        if is_dataclass(obj):

    # Special handling for objects containing image arrays
            if hasattr(obj, "file_path"):
                return str(obj.file_path).replace("\\", "/")

            return {
                field.name: self._serialize(getattr(obj, field.name))
                for field in fields(obj)
            }

        if isinstance(obj, dict):
            return {
            key: self._serialize(value)
            for key, value in obj.items()
        }

        if isinstance(obj, list):
            return [
            self._serialize(item)
            for item in obj
        ]

        if isinstance(obj, tuple):
            return [
            self._serialize(item)
            for item in obj
        ]

        if isinstance(obj, Enum):
            return obj.value

        if isinstance(obj, Path):
            return str(obj)

        if isinstance(obj, datetime):
            return obj.isoformat()

    # Don't send huge images in JSON
        if isinstance(obj, np.ndarray):
            return None

        return obj


    def to_dict(
    self,
    report: ForensicReport,
) -> dict[str, Any]:

        if report is None:
            raise InvalidReportError(
            "Report cannot be None."
        )

        return self._serialize(report)
    def to_json(
        self,
        report: ForensicReport,
        *,
        indent: int = 4,
    ) -> str:
        """
        Convert a forensic report to JSON.
        """

        dictionary = self.to_dict(report)

        return json.dumps(
            dictionary,
            indent=indent,
            default=str,
        )
        
    def save_json(
        self,
        report: ForensicReport,
        output_path: str | Path,
    ) -> Path:
        """
        Save report as JSON.
        """

        output_path = Path(output_path)

        output_path.write_text(
            self.to_json(report),
            encoding="utf-8",
        )

        return output_path
    def pretty_print(
        self,
        report: ForensicReport,
    ) -> None:
        """
        Print a formatted report.
        """

        print(
            self.to_json(
                report,
                indent=4,
            )
        )