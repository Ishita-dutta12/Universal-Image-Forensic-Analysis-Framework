"""
UIFAF Visualization Data Models

This module defines the dataclasses used throughout the
Visualization & Localization module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ============================================================
# Bounding Box
# ============================================================

@dataclass(slots=True)
class BoundingBox:
    """
    Represents one localized suspicious region.
    """

    x: int
    y: int
    width: int
    height: int
    confidence: float


# ============================================================
# Heatmap Result
# ============================================================

@dataclass(slots=True)
class HeatmapResult:
    """
    Stores the generated heatmap.
    """

    heatmap: np.ndarray
    file_path: Optional[Path] = None


# ============================================================
# Overlay Result
# ============================================================

@dataclass(slots=True)
class OverlayResult:
    """
    Stores the overlay image.
    """

    overlay: np.ndarray
    file_path: Optional[Path] = None


# ============================================================
# Localization Result
# ============================================================


@dataclass(slots=True)
class LocalizationResult:
    """
    Complete visualization output.
    """

    heatmap: HeatmapResult

    overlay: OverlayResult

    binary_mask: Optional[np.ndarray] = None

    suspicious_regions: list[BoundingBox] = field(default_factory=list)

    localization_confidence: float = 0.0

    explanation_method: str = "Feature Activation Map"

    success: bool = True

    message: str = "Localization completed successfully."

    # -------------------------------
    # Backward compatibility
    # -------------------------------

    @property
    def bounding_boxes(self):
        return self.suspicious_regions

    @property
    def confidence(self):
        return self.localization_confidence