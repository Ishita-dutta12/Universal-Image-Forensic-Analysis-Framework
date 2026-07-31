"""
UIFAF Overlay Generator

Blends the original image with the generated heatmap
to create an interpretable forensic visualization.
"""

from __future__ import annotations

import cv2
import numpy as np

from .constants import OVERLAY_ALPHA
from .exceptions import OverlayGenerationError
from .localization_models import OverlayResult


class OverlayGenerator:
    """
    Creates visualization overlays.
    """

    def __init__(
        self,
        alpha: float = OVERLAY_ALPHA,
    ) -> None:
        """
        Parameters
        ----------
        alpha : float
            Heatmap blending ratio.
            Must lie between 0 and 1.
        """

        if not 0.0 <= alpha <= 1.0:
            raise OverlayGenerationError(
                "Overlay alpha must be between 0 and 1."
            )

        self.alpha = alpha

    def generate(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
    ) -> OverlayResult:
        """
        Blend heatmap with original RGB image.

        Parameters
        ----------
        image : np.ndarray
            Original RGB image.

        heatmap : np.ndarray
            RGB heatmap.

        Returns
        -------
        OverlayResult
        """

        try:

            if image.shape != heatmap.shape:
                raise OverlayGenerationError(
                    "Image and heatmap must have identical dimensions."
                )

            if image.dtype != np.uint8:
                image = image.astype(np.uint8)

            if heatmap.dtype != np.uint8:
                heatmap = heatmap.astype(np.uint8)

            overlay = cv2.addWeighted(
                image,
                1.0 - self.alpha,
                heatmap,
                self.alpha,
                0,
            )

            return OverlayResult(
                overlay=overlay,
            )

        except Exception as exc:

            raise OverlayGenerationError(
                f"Overlay generation failed: {exc}"
            ) from exc