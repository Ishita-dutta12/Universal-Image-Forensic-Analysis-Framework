"""
Custom exceptions for the Report Generator module.
"""


class ReportGeneratorError(Exception):
    """Base exception for report generator."""
    pass


class ReportGenerationError(ReportGeneratorError):
    """Raised when report generation fails."""
    pass


class InvalidReportError(ReportGeneratorError):
    """Raised when a forensic report is invalid."""
    pass


class ReportValidationError(ReportGeneratorError):
    """Raised when a generated report is invalid."""
    pass