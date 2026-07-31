"""
UIFAF Visualization Engine

Coordinates Grad-CAM, heatmap generation,
overlay creation, and saving visualization outputs.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .gradcam import GradCAM
from .heatmap import HeatmapGenerator
from .localization_models import LocalizationResult
from .overlay import OverlayGenerator


class VisualizationEngine:
    """
    Complete visualization pipeline.

    Pipeline
    --------
    Image
        ↓
    GradCAM
        ↓
    Heatmap
        ↓
    Overlay
        ↓
    Save Images
        ↓
    LocalizationResult
    """

    def __init__(
        self,
        model: torch.nn.Module,
        
        output_directory: Path | str = "outputs",
    ) -> None:

        self.gradcam = GradCAM(
            model=model,
          
        )

        self.heatmap_generator = HeatmapGenerator()

        self.overlay_generator = OverlayGenerator()

        self.output_directory = Path(output_directory)

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def generate(
        self,
        image_tensor: torch.Tensor,
        original_image: np.ndarray,
        class_index: int | None = None,
    ) -> LocalizationResult:
        """
        Generate complete visualization results.

        Parameters
        ----------
        image_tensor
            Model input tensor.

        original_image
            Original RGB image as numpy array.

        class_index
            Target class.
            If None, highest-scoring class is used.

        Returns
        -------
        LocalizationResult
        """

        # ---------------------------------------------------
        # GradCAM
        # ---------------------------------------------------

        cam = self.gradcam.generate(
            image_tensor=image_tensor,
            
        )

        # ---------------------------------------------------
        # Heatmap
        # ---------------------------------------------------

        heatmap_result = self.heatmap_generator.generate(cam)

        # ---------------------------------------------------
        # Overlay
        # ---------------------------------------------------

        overlay_result = self.overlay_generator.generate(
            image=original_image,
            heatmap=heatmap_result.heatmap,
        )

        # ---------------------------------------------------
        # Save Heatmap
        # ---------------------------------------------------

        heatmap_path = (
            self.output_directory / "heatmap.png"
        )

        cv2.imwrite(
            str(heatmap_path),
            cv2.cvtColor(
                heatmap_result.heatmap,
                cv2.COLOR_RGB2BGR,
            ),
        )

        heatmap_result.file_path = heatmap_path

        # ---------------------------------------------------
        # Save Overlay
        # ---------------------------------------------------

        overlay_path = (
            self.output_directory / "overlay.png"
        )

        cv2.imwrite(
            str(overlay_path),
            cv2.cvtColor(
                overlay_result.overlay,
                cv2.COLOR_RGB2BGR,
            ),
        )

        overlay_result.file_path = overlay_path

        # ---------------------------------------------------
        # Build Localization Result
        # ---------------------------------------------------

        localization = LocalizationResult(
            heatmap=heatmap_result,
            overlay=overlay_result,
            binary_mask=None,
            suspicious_regions=[],
            localization_confidence=float(cam.mean()),
            explanation_method="Feature Activation Map",
            success=True,
            message="Visualization completed successfully.",
        )

        return localization

    def close(self) -> None:
        """
        Release Grad-CAM hooks.
        """

        self.gradcam.remove_hooks()