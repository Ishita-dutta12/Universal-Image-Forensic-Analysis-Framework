"""
UIFAF Pipeline Validator

Validates requests and pipeline state before execution.

Author: Ishita Dutta
Project: UIFAF
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from httpcore import request

from .exceptions import (
    ConfigurationError,
    PipelineValidationError,
)
from .pipeline_models import (
    PipelineRequest,
)


class PipelineValidator:
    """
    Validator for the pipeline module.
    """

    SUPPORTED_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }

    @staticmethod
    def validate_request(
        request: PipelineRequest,
    ) -> None:
        """
        Validate pipeline request.
        """
        print("=" * 60)
        print("DEBUG: validate_request()")
        print("request =", request)
        print("type(request) =", type(request))
        print("PipelineRequest =", PipelineRequest)
        print("type(request) is PipelineRequest =", type(request) is PipelineRequest)
        print("isinstance =", isinstance(request, PipelineRequest))
        print("=" * 60)
        if not isinstance(
            request,
            PipelineRequest,
        ):
            raise PipelineValidationError(
                "Invalid PipelineRequest."
            )

        image_path = request.image_path

        if not isinstance(image_path, Path):
            raise PipelineValidationError(
                "image_path must be Path."
            )

        if not image_path.exists():
            raise PipelineValidationError(
                f"Image not found: {image_path}"
            )

        if image_path.suffix.lower() not in (
            PipelineValidator.SUPPORTED_EXTENSIONS
        ):
            raise PipelineValidationError(
                f"Unsupported image format: "
                f"{image_path.suffix}"
            )

    @staticmethod
    def validate_module(
        module: Any,
        module_name: str,
    ) -> None:
        """
        Ensure a required module exists.
        """

        if module is None:
            raise ConfigurationError(
                f"{module_name} has not been initialized."
            )

    @staticmethod
    def validate_modules(
        **modules: Any,
    ) -> None:
        """
        Validate all required modules.
        """

        for name, module in modules.items():

            PipelineValidator.validate_module(
                module,
                name,
            )