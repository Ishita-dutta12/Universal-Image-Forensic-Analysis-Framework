"""
Validation utilities for the Universal Forensic Reasoning Engine (UFRE).

Provides centralized validation routines used across reasoning modules.

Author: UIFAF
"""

from __future__ import annotations

import torch
from torch import Tensor

from representation.representation_models import (
    RepresentationOutput,
)

from ufre.ufre_models import (
    ConfidenceEstimate,
    EvidenceFeature,
    StageEvidence,
)

from ufre.exceptions import (
    InvalidUFREInputError,
    InvalidUFREOutputError,
)


class ReasoningValidator:
    """
    Collection of reusable validation utilities for UFRE.
    """

    @staticmethod
    def validate_representation_output(
        representations: RepresentationOutput,
    ) -> None:
        """
        Validate a RepresentationOutput object.
        """

        if not isinstance(
            representations,
            RepresentationOutput,
        ):
            raise InvalidUFREInputError(
                f"Expected RepresentationOutput, "
                f"got {type(representations).__name__}."
            )

        representations.validate()

    @staticmethod
    def validate_stage_evidence(
        stage: StageEvidence,
    ) -> None:
        """
        Validate a StageEvidence object.
        """

        if not isinstance(stage, StageEvidence):
            raise InvalidUFREOutputError(
                f"Expected StageEvidence, "
                f"got {type(stage).__name__}."
            )

        stage.validate()

    @staticmethod
    def validate_stage_collection(
        stages: tuple[StageEvidence, ...],
    ) -> None:
        """
        Validate a collection of StageEvidence objects.

        Parameters
        ----------
        stages
            Tuple of stage evidence.

        Raises
        ------
        InvalidUFREOutputError
            If the collection is invalid.
        """

        if not isinstance(stages, tuple):
            raise InvalidUFREOutputError(
                "Stage collection must be a tuple."
            )

        if len(stages) != 4:
            raise InvalidUFREOutputError(
                f"Expected 4 stages, got {len(stages)}."
            )

        for stage in stages:
            ReasoningValidator.validate_stage_evidence(stage)

    @staticmethod
    def validate_evidence_feature(
        evidence: EvidenceFeature,
    ) -> None:
        """
        Validate an EvidenceFeature object.
        """

        if not isinstance(evidence, EvidenceFeature):
            raise InvalidUFREOutputError(
                f"Expected EvidenceFeature, got "
                f"{type(evidence).__name__}."
            )

        evidence.validate()

    @staticmethod
    def validate_confidence(
        confidence: ConfidenceEstimate,
    ) -> None:
        """
        Validate a ConfidenceEstimate.
        """

        if not isinstance(
            confidence,
            ConfidenceEstimate,
        ):
            raise InvalidUFREOutputError(
                f"Expected ConfidenceEstimate, got "
                f"{type(confidence).__name__}."
            )

        confidence.validate()
    @staticmethod
    def validate_same_shape(
        tensor1: Tensor,
        tensor2: Tensor,
    ) -> None:
        """
        Validate that two tensors have identical shapes.

        Raises
        ------
        InvalidUFREInputError
            If tensor shapes differ.
        """

        if tensor1.shape != tensor2.shape:
            raise InvalidUFREInputError(
                f"Shape mismatch: "
                f"{tensor1.shape} != {tensor2.shape}."
            )

    @staticmethod
    def validate_same_device(
        tensor1: Tensor,
        tensor2: Tensor,
    ) -> None:
        """
        Validate that two tensors are located on the same device.

        Raises
        ------
        InvalidUFREInputError
            If tensor devices differ.
        """

        if tensor1.device != tensor2.device:
            raise InvalidUFREInputError(
                f"Device mismatch: "
                f"{tensor1.device} != {tensor2.device}."
            )

    @staticmethod
    def validate_same_dtype(
        tensor1: Tensor,
        tensor2: Tensor,
    ) -> None:
        """
        Validate that two tensors have the same dtype.

        Raises
        ------
        InvalidUFREInputError
            If tensor dtypes differ.
        """

        if tensor1.dtype != tensor2.dtype:
            raise InvalidUFREInputError(
                f"Dtype mismatch: "
                f"{tensor1.dtype} != {tensor2.dtype}."
            )

    @staticmethod
    def validate_finite(
        tensor: Tensor,
    ) -> None:
        """
        Validate that a tensor contains only finite values.

        Raises
        ------
        InvalidUFREInputError
            If the tensor contains NaN or Inf values.
        """

        if not isinstance(tensor, Tensor):
            raise InvalidUFREInputError(
                f"Expected torch.Tensor, got "
                f"{type(tensor).__name__}."
            )

        if not torch.isfinite(tensor).all():
            raise InvalidUFREInputError(
                "Tensor contains NaN or Inf values."
            )