
"""
Representation Validator.

Validates the complete representation output before
it is consumed by downstream modules.

Author: Ishita Dutta
Project: UIFAF
"""

from __future__ import annotations

import torch

from .exceptions import (
    InvalidRepresentationOutputError,
)

from .representation_models import (
    RepresentationOutput,
)


class RepresentationValidator:
    """
    Performs structural validation of the complete
    RepresentationOutput.
    """

    REQUIRED_STAGES = (
        "stage1",
        "stage2",
        "stage3",
        "stage4",
    )

    @classmethod
    def validate(
        cls,
        output: RepresentationOutput,
    ) -> None:
        """
        Validate the complete representation output.
        """

        output.validate()

        if output.stage_names != cls.REQUIRED_STAGES:
            raise InvalidRepresentationOutputError(
                "Invalid stage ordering."
            )

        for stage in output.stages:

            stage.validate()

            cls._validate_shapes(stage)

            cls._validate_batch(stage)

            cls._validate_dtype(stage)

            cls._validate_finite(stage)
    @staticmethod
    def _validate_shapes(stage):

        spatial = stage.spatial.tensor.shape
        frequency = stage.frequency.tensor.shape
        context = stage.context.tensor.shape
        fused = stage.fused.tensor.shape

        if spatial != frequency:
            raise InvalidRepresentationOutputError(
                f"{stage.stage}: spatial/frequency shape mismatch."
            )

        if spatial != context:
            raise InvalidRepresentationOutputError(
                f"{stage.stage}: spatial/context shape mismatch."
            )

        if spatial != fused:
            raise InvalidRepresentationOutputError(
                f"{stage.stage}: fused shape mismatch."
            )
    @staticmethod
    def _validate_batch(stage):

        batch = stage.spatial.tensor.shape[0]

        if stage.frequency.tensor.shape[0] != batch:
            raise InvalidRepresentationOutputError(
                f"{stage.stage}: frequency batch mismatch."
            )

        if stage.context.tensor.shape[0] != batch:
            raise InvalidRepresentationOutputError(
                f"{stage.stage}: context batch mismatch."
            )

        if stage.fused.tensor.shape[0] != batch:
            raise InvalidRepresentationOutputError(
                f"{stage.stage}: fused batch mismatch."
            )
    @staticmethod
    def _validate_dtype(stage):

        dtype = stage.spatial.tensor.dtype

        if stage.frequency.tensor.dtype != dtype:
            raise InvalidRepresentationOutputError(
                f"{stage.stage}: dtype mismatch."
            )

        if stage.context.tensor.dtype != dtype:
            raise InvalidRepresentationOutputError(
                f"{stage.stage}: dtype mismatch."
            )

        if stage.fused.tensor.dtype != dtype:
            raise InvalidRepresentationOutputError(
                f"{stage.stage}: dtype mismatch."
            )
    @staticmethod
    def _validate_finite(stage):

        for feature in (
            stage.spatial,
            stage.frequency,
            stage.context,
            stage.fused,
        ):

            if not torch.isfinite(feature.tensor).all():

                raise InvalidRepresentationOutputError(
                    f"{stage.stage}: tensor contains NaN or Inf."
                )