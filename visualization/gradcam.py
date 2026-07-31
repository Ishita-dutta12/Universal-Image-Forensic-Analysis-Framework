"""
UIFAF Feature Heatmap Generator

Generates a visualization heatmap directly from the
final backbone feature map.

This replaces classical Grad-CAM because UIFAF uses
a feature extraction backbone instead of a classifier.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .exceptions import HeatmapGenerationError
from .visualization_validator import VisualizationValidator


class GradCAM:
    """
    Feature-map visualization.

    Although the class is still called GradCAM
    (to avoid changing the project architecture),
    this implementation produces a localization
    heatmap directly from the final backbone
    feature map.
    """

    def __init__(
        self,
        model: torch.nn.Module,
    ) -> None:

        VisualizationValidator.validate_model(model)

        self.model = model

    def generate(
        self,
        image_tensor: torch.Tensor,
    ) -> np.ndarray:
        """
        Generate feature activation heatmap.

        Parameters
        ----------
        image_tensor
            Normalized model input.

        Returns
        -------
        np.ndarray
            Normalized heatmap in range [0,1].
        """

        try:

            with torch.no_grad():

                outputs = self.model(image_tensor)

            # FeatureListNet returns:
            # [stage1, stage2, stage3, stage4]

            if isinstance(outputs, (list, tuple)):

                feature_map = outputs[-1]

            else:

                feature_map = outputs

            # ----------------------------
            # Average activation map
            # ----------------------------

            heatmap = feature_map.mean(dim=1)

            heatmap = F.relu(heatmap)

            heatmap = heatmap.squeeze()

            heatmap = heatmap.cpu().numpy()

            # ----------------------------
            # Normalize
            # ----------------------------

            heatmap -= heatmap.min()

            heatmap /= (heatmap.max() + 1e-8)

            # ----------------------------
            # Resize
            # ----------------------------

            height = image_tensor.shape[2]

            width = image_tensor.shape[3]

            heatmap = cv2.resize(

                heatmap,

                (width, height),

                interpolation=cv2.INTER_CUBIC,

            )

            return heatmap

        except Exception as exc:

            raise HeatmapGenerationError(

                f"Heatmap generation failed: {exc}"

            ) from exc

    def remove_hooks(self) -> None:
        """
        Kept only for compatibility.

        No hooks are used.
        """

        return