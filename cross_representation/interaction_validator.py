"""
Cross Representation Validation Utilities.

This module centralizes all validation logic used by the
cross-representation interaction modules.

Author: Ishita Dutta
Project: UIFAF
"""

from __future__ import annotations

from typing import Optional

import torch

from representation.representation_models import (
    RepresentationFeature,
    StageRepresentation,
)

from .exceptions import (
    ChannelMismatchError,
    DeviceMismatchError,
    RepresentationMismatchError,
    ShapeMismatchError,
)


class InteractionValidator:
    """
    Utility class for validating representation consistency.

    All methods are static and raise descriptive exceptions
    when inconsistencies are detected.
    """

    @staticmethod
    def _available_features(
        stage: StageRepresentation,
    ) -> list[RepresentationFeature]:
        """
        Return all non-None representations.
        """
        return [
            feature
            for feature in (
                stage.spatial,
                stage.frequency,
                stage.context,
                stage.fused,
            )
            if feature is not None
        ]
    @staticmethod
    def validate_shape_consistency(
        stage: StageRepresentation,
    ) -> None:
        """
        Ensure all available representations have identical shapes.
        """
        features = InteractionValidator._available_features(stage)

        if len(features) <= 1:
            return

        reference_shape = features[0].tensor.shape

        for feature in features[1:]:
            if feature.tensor.shape != reference_shape:
                raise ShapeMismatchError(
                    expected=reference_shape,
                    received=feature.tensor.shape,
                )

    @staticmethod
    def validate_device_consistency(
        stage: StageRepresentation,
    ) -> None:
        """
        Ensure all representations are located on the same device.
        """
        features = InteractionValidator._available_features(stage)

        if len(features) <= 1:
            return

        reference_device = features[0].tensor.device

        for feature in features[1:]:
            if feature.tensor.device != reference_device:
                raise DeviceMismatchError(
                    expected=str(reference_device),
                    received=str(feature.tensor.device),
                )

    @staticmethod
    def validate_dtype_consistency(
        stage: StageRepresentation,
    ) -> None:
        """
        Ensure all representations use the same tensor dtype.
        """
        features = InteractionValidator._available_features(stage)

        if len(features) <= 1:
            return

        reference_dtype = features[0].tensor.dtype

        for feature in features[1:]:
            if feature.tensor.dtype != reference_dtype:
                raise RepresentationMismatchError(
                    "All representations must have identical tensor dtypes."
                )
    @staticmethod
    def validate_channel_consistency(
        stage: StageRepresentation,
    ) -> None:
        """
        Ensure all representations have identical channel dimensions.
        """
        features = InteractionValidator._available_features(stage)

        if len(features) <= 1:
            return

        reference_channels = features[0].tensor.shape[1]

        for feature in features[1:]:
            if feature.tensor.shape[1] != reference_channels:
                raise ChannelMismatchError(
                    expected=reference_channels,
                    received=feature.tensor.shape[1],
                )

    @staticmethod
    def validate_batch_consistency(
        stage: StageRepresentation,
    ) -> None:
        """
        Ensure all representations belong to the same batch size.
        """
        features = InteractionValidator._available_features(stage)

        if len(features) <= 1:
            return

        reference_batch = features[0].tensor.shape[0]

        for feature in features[1:]:
            if feature.tensor.shape[0] != reference_batch:
                raise RepresentationMismatchError(
                    "All representations must have identical batch sizes."
                )

    @staticmethod
    def validate_spatial_consistency(
        stage: StageRepresentation,
    ) -> None:
        """
        Ensure all representations have identical spatial dimensions.
        """
        features = InteractionValidator._available_features(stage)

        if len(features) <= 1:
            return

        reference_size = features[0].tensor.shape[-2:]

        for feature in features[1:]:
            if feature.tensor.shape[-2:] != reference_size:
                raise ShapeMismatchError(
                    expected=reference_size,
                    received=feature.tensor.shape[-2:],
                )
    @staticmethod
    def validate_finite(
        stage: StageRepresentation,
    ) -> None:
        """
        Ensure all tensors contain only finite values.
        """
        for feature in InteractionValidator._available_features(stage):
            if not torch.isfinite(feature.tensor).all():
                raise RepresentationMismatchError(
                    f"Stage '{stage.stage}' contains NaN or Inf values "
                    f"in '{feature.representation_type}'."
                )

    @staticmethod
    def validate_stage(
        stage: StageRepresentation,
    ) -> None:
        """
        Perform all consistency checks for a stage.
        """
        InteractionValidator.validate_shape_consistency(stage)
        InteractionValidator.validate_device_consistency(stage)
        InteractionValidator.validate_dtype_consistency(stage)
        InteractionValidator.validate_channel_consistency(stage)
        InteractionValidator.validate_batch_consistency(stage)
        InteractionValidator.validate_spatial_consistency(stage)
        InteractionValidator.validate_finite(stage)
    