"""
Abstract base class for Cross Representation Interaction.

Every interaction algorithm (Cross Attention, Gated Interaction,
etc.) inherits from this class.

Author: Ishita Dutta
Project: UIFAF
"""

from __future__ import annotations

from abc import ABC, abstractmethod
#from pyexpat import features
from typing import Final

import torch
import torch.nn as nn
from .exceptions import DeviceMismatchError
from representation.representation_models import (
    RepresentationFeature,
    RepresentationOutput,
    StageRepresentation,
)

from .exceptions import (
    CrossRepresentationError,
    
    
  
    InvalidInteractionInputError,
    InvalidInteractionOutputError,
    ShapeMismatchError,
    StageInteractionError,
)


class BaseInteraction(nn.Module, ABC):
    """
    Abstract base class for all interaction modules.

    Pipeline
    --------
    RepresentationOutput
            │
            ▼
        validate()
            │
            ▼
      process_stage()
            │
            ▼
    RepresentationOutput
            │
            ▼
        validate()
            │
            ▼
          return
    """

    STAGES: Final[tuple[str, ...]] = (
        "stage1",
        "stage2",
        "stage3",
        "stage4",
    )

    def __init__(self) -> None:
        super().__init__()

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def forward(
        self,
        representation: RepresentationOutput,
    ) -> RepresentationOutput:
        """
        Run cross-representation interaction.

        Parameters
        ----------
        representation:
            Output produced by the Representation Module.

        Returns
        -------
        RepresentationOutput
            Enhanced representation output.
        """

        self._validate_input(representation)

        stage1 = self._process_stage(representation.stage1)
        stage2 = self._process_stage(representation.stage2)
        stage3 = self._process_stage(representation.stage3)
        stage4 = self._process_stage(representation.stage4)

        output = RepresentationOutput(
            stage1=stage1,
            stage2=stage2,
            stage3=stage3,
            stage4=stage4,
            original_size=representation.original_size,
        )

        self._validate_output(output)

        return output

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def _validate_input(
        self,
        representation: RepresentationOutput,
    ) -> None:
        """
        Validate input representation.
        """

        if not isinstance(
            representation,
            RepresentationOutput,
        ):
            raise InvalidInteractionInputError(
                "Input must be a RepresentationOutput."
            )

        representation.validate()

    def _validate_output(
        self,
        output: RepresentationOutput,
    ) -> None:
        """
        Validate generated output.
        """

        if not isinstance(
            output,
            RepresentationOutput,
        ):
            raise InvalidInteractionOutputError(
                "Output must be a RepresentationOutput."
            )

        output.validate()

    # ---------------------------------------------------------
    # Stage Processing
    # ---------------------------------------------------------

    def _process_stage(
        self,
        stage: StageRepresentation,
    ) -> StageRepresentation:
    

        try:
            return self.interact_stage(stage)

        except CrossRepresentationError:
            raise

        except Exception as exc:
            raise StageInteractionError(
                stage.stage,
                str(exc),
            ) from exc
            
        # ---------------------------------------------------------
    # Abstract Interface
    # ---------------------------------------------------------

    @abstractmethod
    def interact_stage(
        self,
        stage: StageRepresentation,
    ) -> StageRepresentation:
        """
        Perform cross-representation interaction
        for one backbone stage.

        Subclasses must return a new validated
        StageRepresentation.

        Parameters
        ----------
        stage:
            Stage representations from Module 5.

        Returns
        -------
        StageRepresentation
            Enhanced stage representation.
        """
        raise NotImplementedError(
        "Subclasses must implement interact_stage()."
        )

    # ---------------------------------------------------------
    # Helper Methods
    # ---------------------------------------------------------
    # ---------------------------------------------------------
# Helper Methods
# ---------------------------------------------------------

    @staticmethod
    def available_features(
        stage: StageRepresentation,
    ) -> tuple[RepresentationFeature, ...]:

        features = [stage.spatial]

        if stage.frequency is not None:
            features.append(stage.frequency)

        if stage.context is not None:
            features.append(stage.context)

        if stage.fused is not None:
            features.append(stage.fused)

        return tuple(features)



    @staticmethod
    def clone_feature(
        feature: RepresentationFeature,
        tensor: torch.Tensor,
    ) -> RepresentationFeature:

        new_feature = RepresentationFeature(
            stage=feature.stage,
            tensor=tensor,
            representation_type=feature.representation_type,
        )

        new_feature.validate()

        return new_feature

    
    @staticmethod
    def check_same_shape(
        *features: RepresentationFeature | None,
    ) -> None:

        valid_features = [
            feature
            for feature in features
            if feature is not None
        ]

        if len(valid_features) <= 1:
            return

        shape = valid_features[0].tensor.shape

        for feature in valid_features[1:]:
            if feature.tensor.shape != shape:
                raise ShapeMismatchError(
                    shape,
                    feature.tensor.shape,
                )

    
    @staticmethod
    def check_same_device(
        *features: RepresentationFeature | None,
    ) -> None:

        valid_features = [
            feature
            for feature in features
            if feature is not None
        ]

        if len(valid_features) <= 1:
            return

        device = valid_features[0].tensor.device

        for feature in valid_features[1:]:
            if feature.tensor.device != device:
                raise DeviceMismatchError(
                    str(device),
                    str(feature.tensor.device),
                )

    
    @staticmethod
    def check_same_dtype(
        *features: RepresentationFeature | None,
    ) -> None:

        valid_features = [
            feature
            for feature in features
            if feature is not None
        ]

        if len(valid_features) <= 1:
            return

        dtype = valid_features[0].tensor.dtype

        for feature in valid_features[1:]:
            if feature.tensor.dtype != dtype:
                raise CrossRepresentationError(
                    "Representation tensors have different dtypes."
                )

    @staticmethod
    def ensure_finite(
        tensor: torch.Tensor,
    ) -> None:
        """
        Ensure tensor contains only finite values.
        """

        if not torch.isfinite(tensor).all():
            raise CrossRepresentationError(
                "Interaction produced NaN or Inf values."
            )

    def extra_repr(self) -> str:
        return self.__class__.__name__
