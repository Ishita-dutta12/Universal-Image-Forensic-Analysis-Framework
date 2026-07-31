from pathlib import Path

import numpy as np
import pytest

from visualization.exceptions import VisualizationValidationError
from visualization.visualization_validator import VisualizationValidator


def test_validate_rgb_image():
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    VisualizationValidator.validate_image(image)


def test_invalid_image_channels():
    image = np.zeros((224, 224), dtype=np.uint8)

    with pytest.raises(VisualizationValidationError):
        VisualizationValidator.validate_image(image)


def test_invalid_image_path():
    with pytest.raises(VisualizationValidationError):
        VisualizationValidator.validate_image_path(
            Path("does_not_exist.jpg")
        )


def test_invalid_model():
    with pytest.raises(Exception):
        VisualizationValidator.validate_model(None)


def test_invalid_explainer():
    with pytest.raises(VisualizationValidationError):
        VisualizationValidator.validate_explainer("unknowncam")