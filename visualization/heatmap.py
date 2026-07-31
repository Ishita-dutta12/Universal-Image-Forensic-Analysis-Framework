"""
UIFAF Heatmap Generator

Converts normalized Grad-CAM activation maps into
colored heatmaps suitable for visualization.
"""

from __future__ import annotations

import cv2
import numpy as np

from .constants import (
    HEATMAP_COLORMAP,
)
from .exceptions import (
    HeatmapGenerationError,
)
from .localization_models import HeatmapResult


class HeatmapGenerator:
    """
    Generates colored heatmaps from Grad-CAM outputs.
    """

    _COLORMAPS = {
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "bone": cv2.COLORMAP_BONE,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "plasma": cv2.COLORMAP_PLASMA,
        "inferno": cv2.COLORMAP_INFERNO,
        "turbo": cv2.COLORMAP_TURBO,
    }

    def __init__(
        self,
        colormap: str = HEATMAP_COLORMAP,
    ) -> None:

        if colormap.lower() not in self._COLORMAPS:
            raise HeatmapGenerationError(
                f"Unsupported colormap '{colormap}'."
            )

        self.colormap = self._COLORMAPS[colormap.lower()]

    def generate(
        self,
        cam: np.ndarray,
    ) -> HeatmapResult:
        """
        Convert a normalized CAM into a colored heatmap.

        Parameters
        ----------
        cam : np.ndarray
            Normalized Grad-CAM output in range [0,1].

        Returns
        -------
        HeatmapResult
        """

        try:

            if cam.ndim != 2:
                raise HeatmapGenerationError(
                    "Grad-CAM output must be a 2D array."
                )

            cam = np.clip(cam, 0.0, 1.0)

            cam_uint8 = np.uint8(cam * 255)

            heatmap = cv2.applyColorMap(
                cam_uint8,
                self.colormap,
            )

            heatmap = cv2.cvtColor(
                heatmap,
                cv2.COLOR_BGR2RGB,
            )

            return HeatmapResult(
                heatmap=heatmap,
            )

        except Exception as exc:

            raise HeatmapGenerationError(
                f"Heatmap generation failed: {exc}"
            ) from exc