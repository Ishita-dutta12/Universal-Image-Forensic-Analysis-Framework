"""
Exceptions for the Decision Engine.

Author: Ishita Dutta
Project: UIFAF
"""


class DecisionError(Exception):
    """Base exception for Decision Engine."""


class InvalidDecisionInputError(DecisionError):
    """Raised when Decision Engine receives invalid inputs."""


class ClassificationError(DecisionError):
    """Raised when classification fails."""


class EvidenceFusionError(DecisionError):
    """Raised when evidence fusion fails."""


class RiskAssessmentError(DecisionError):
    """Raised when risk assessment fails."""


class ConfidenceAssessmentError(DecisionError):
    """Raised when confidence estimation fails."""