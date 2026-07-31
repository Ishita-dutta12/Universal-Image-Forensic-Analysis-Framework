"""
UIFAF Visualization Validator

Validates all inputs required for the Visualization &
Localization module before Grad-CAM execution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .constants import (
    DEFAULT_EXPLAINER,
    HEATMAP_DIRECTORY,
    MASK_DIRECTORY,
    OVERLAY_DIRECTORY,
    SUPPORTED_EXPLAINERS,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from .exceptions import (
    InvalidModelError,
    InvalidTargetLayerError,
    VisualizationValidationError,
)


class VisualizationValidator:
    """
    Validator for the visualization module.
    """

    @staticmethod
    def validate_image(image: np.ndarray) -> None:
        """
        Validate an RGB image.

        Raises
        ------
        VisualizationValidationError
        """

        if not isinstance(image, np.ndarray):
            raise VisualizationValidationError(
                "Image must be a NumPy array."
            )

        if image.ndim != 3:
            raise VisualizationValidationError(
                "Image must be a 3-dimensional array."
            )

        if image.shape[2] != 3:
            raise VisualizationValidationError(
                "Image must contain exactly 3 RGB channels."
            )

    @staticmethod
    def validate_image_path(image_path: Path) -> None:
        """
        Validate image path.
        """

        if not isinstance(image_path, Path):
            image_path = Path(image_path)

        if not image_path.exists():
            raise VisualizationValidationError(
                f"Image not found: {image_path}"
            )

        if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise VisualizationValidationError(
                f"Unsupported image format: {image_path.suffix}"
            )

    @staticmethod
    def validate_model(model: Any) -> None:
        """
        Validate model instance.
        """

        if model is None:
            raise InvalidModelError(
                "Model cannot be None."
            )

    @staticmethod
    def validate_target_layer(target_layer: str) -> None:
        """
        Validate Grad-CAM target layer.
        """

        if not isinstance(target_layer, str):
            raise InvalidTargetLayerError(
                "Target layer must be a string."
            )

        if not target_layer.strip():
            raise InvalidTargetLayerError(
                "Target layer cannot be empty."
            )

    @staticmethod
    def validate_explainer(explainer: str) -> None:
        """
        Validate explanation method.
        """

        if explainer not in SUPPORTED_EXPLAINERS:
            raise VisualizationValidationError(
                f"Unsupported explainer '{explainer}'. "
                f"Supported: {SUPPORTED_EXPLAINERS}"
            )

    @staticmethod
    def validate_heatmap(
        heatmap: np.ndarray,
        image: np.ndarray,
    ) -> None:
        """
        Ensure heatmap matches image resolution.
        """

        if heatmap.shape[:2] != image.shape[:2]:
            raise VisualizationValidationError(
                "Heatmap size does not match image size."
            )

    @staticmethod
    def create_output_directories() -> None:
        """
        Create visualization output directories.
        """

        Path(HEATMAP_DIRECTORY).mkdir(
            parents=True,
            exist_ok=True,
        )

        Path(OVERLAY_DIRECTORY).mkdir(
            parents=True,
            exist_ok=True,
        )

        Path(MASK_DIRECTORY).mkdir(
            parents=True,
            exist_ok=True,
        )

    @classmethod
    def validate(
        cls,
        *,
        image: np.ndarray,
        image_path: Path,
        model: Any,
        target_layer: str,
        explainer: str = DEFAULT_EXPLAINER,
    ) -> None:
        """
        Perform complete visualization validation.
        """

        cls.validate_image(image)

        cls.validate_image_path(image_path)

        cls.validate_model(model)

        cls.validate_target_layer(target_layer)

        cls.validate_explainer(explainer)

        cls.create_output_directories()