"""
UIFAF Pipeline Exceptions

Defines custom exceptions used by the pipeline module.
"""

from __future__ import annotations


class PipelineError(Exception):
    """
    Base exception for the pipeline.
    """

    def __init__(
        self,
        message: str = "Pipeline execution failed.",
    ) -> None:
        super().__init__(message)


class PipelineValidationError(PipelineError):
    """
    Raised when pipeline validation fails.
    """

    def __init__(
        self,
        message: str = "Pipeline validation failed.",
    ) -> None:
        super().__init__(message)


class ModuleExecutionError(PipelineError):
    """
    Raised when one of the UIFAF modules fails.
    """

    def __init__(
        self,
        module_name: str,
        message: str,
    ) -> None:

        super().__init__(
            f"[{module_name}] {message}"
        )

        self.module_name = module_name


class ConfigurationError(PipelineError):
    """
    Raised for invalid pipeline configuration.
    """

    def __init__(
        self,
        message: str = "Invalid pipeline configuration.",
    ) -> None:
        super().__init__(message)


class UnsupportedModuleError(PipelineError):
    """
    Raised when an unknown module is requested.
    """

    def __init__(
        self,
        module_name: str,
    ) -> None:

        super().__init__(
            f"Unsupported module: {module_name}"
        )

        self.module_name = module_name