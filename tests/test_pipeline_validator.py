from pathlib import Path

import pytest

from pipeline.exceptions import (
    ConfigurationError,
    PipelineValidationError,
)
from pipeline.pipeline_models import (
    PipelineRequest,
)
from pipeline.pipeline_validator import (
    PipelineValidator,
)


def test_invalid_request():

    request = PipelineRequest(
        image_path=Path("missing.jpg")
    )

    with pytest.raises(
        PipelineValidationError
    ):
        PipelineValidator.validate_request(
            request
        )


def test_missing_module():

    with pytest.raises(
        ConfigurationError
    ):
        PipelineValidator.validate_module(
            None,
            "backbone",
        )