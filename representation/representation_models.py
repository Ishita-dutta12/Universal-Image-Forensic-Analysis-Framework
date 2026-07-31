"""
Data models for the Representation Module.

Defines strongly typed containers for spatial,
frequency, and contextual forensic representations.


Project: UIFAF
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .exceptions import (
    InvalidRepresentationFeatureError,
    InvalidRepresentationOutputError,
)


_VALID_STAGES = (
    "stage1",
    "stage2",
    "stage3",
    "stage4",
)

_VALID_REPRESENTATIONS = (
    "spatial",
    "frequency",
    "context",
    "fused",
)


@dataclass(frozen=True)
class RepresentationFeature:
    """
    Represents a single forensic representation
    produced by one encoder.
    """

    stage: str
    tensor: torch.Tensor
    representation_type: str

    def validate(self) -> None:
        """
        Validate the representation feature.
        """

        if self.stage not in _VALID_STAGES:
            raise InvalidRepresentationFeatureError(
                f"Unknown stage '{self.stage}'."
            )

        if self.representation_type not in _VALID_REPRESENTATIONS:
            raise InvalidRepresentationFeatureError(
                f"Unknown representation type "
                f"'{self.representation_type}'."
            )

        if not isinstance(self.tensor, torch.Tensor):
            raise InvalidRepresentationFeatureError(
                "tensor must be a torch.Tensor."
            )

        if self.tensor.ndim != 4:
            raise InvalidRepresentationFeatureError(
                "Representation tensor must be 4-dimensional."
            )

        if not torch.isfinite(self.tensor).all():
            raise InvalidRepresentationFeatureError(
                "Representation tensor contains NaN or Inf."
            )


@dataclass(frozen=True)
class StageRepresentation:
    """
    Complete forensic representations for one
    backbone stage.
    """

    stage: str

    spatial: RepresentationFeature

    frequency: Optional[RepresentationFeature] = None
    context: Optional[RepresentationFeature] = None
    fused: Optional[RepresentationFeature] = None
    def validate(self) -> None:
        """
        Validate one stage representation.
        """

        if self.stage not in _VALID_STAGES:
            raise InvalidRepresentationOutputError(
                f"Unknown stage '{self.stage}'."
            )

        self.spatial.validate()
        
        
          

        if self.spatial.stage != self.stage:
            raise InvalidRepresentationOutputError(
                "Spatial representation stage mismatch."
        )

        if self.spatial.representation_type != "spatial":
            raise InvalidRepresentationOutputError(
                "Invalid spatial representation."
        )

    # Frequency (optional)
        if self.frequency is not None:

            self.frequency.validate()

            if self.frequency.stage != self.stage:
                raise InvalidRepresentationOutputError(
                    "Frequency representation stage mismatch."
            )

            if self.frequency.representation_type != "frequency":
                raise InvalidRepresentationOutputError(
                    "Invalid frequency representation."
            )

    # Context (optional)
        if self.context is not None:

            self.context.validate()

            if self.context.stage != self.stage:
                raise InvalidRepresentationOutputError(
                    "Context representation stage mismatch."
            )

            if self.context.representation_type != "context":
                raise InvalidRepresentationOutputError(
                    "Invalid context representation."
            )

    # Fused (optional)
        if self.fused is not None:

            self.fused.validate()

            if self.fused.stage != self.stage:
                raise InvalidRepresentationOutputError(
                "Fused representation stage mismatch."
            )

            if self.fused.representation_type != "fused":
                raise InvalidRepresentationOutputError(
                "Invalid fused representation."
            )

       
            
@dataclass(frozen=True)
class RepresentationOutput:
    """
    Complete output of the Representation Module.
    """

    stage1: StageRepresentation

    stage2: StageRepresentation

    stage3: StageRepresentation

    stage4: StageRepresentation

    original_size: Tuple[int, int]

    def validate(self) -> None:
        """
        Validate the complete representation output.
        """

        self.stage1.validate()
        self.stage2.validate()
        self.stage3.validate()
        self.stage4.validate()

        if (
            self.stage1.stage != "stage1"
            or self.stage2.stage != "stage2"
            or self.stage3.stage != "stage3"
            or self.stage4.stage != "stage4"
        ):
            raise InvalidRepresentationOutputError(
                "Stage ordering is invalid."
            )

        h, w = self.original_size

        if h <= 0 or w <= 0:
            raise InvalidRepresentationOutputError(
                "Invalid original image size."
            )

    @property
    def stages(self) -> tuple[StageRepresentation, ...]:
        return (
            self.stage1,
            self.stage2,
            self.stage3,
            self.stage4,
        )

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    @property
    def stage_names(self) -> tuple[str, ...]:
        return tuple(
            stage.stage
            for stage in self.stages
        )