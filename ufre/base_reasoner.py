"""
Base class for Universal Forensic Reasoning Engine (UFRE).

Provides the common execution pipeline for stage-wise reasoning modules.

Author: UIFAF
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor

from representation.representation_models import (
    RepresentationOutput,
    StageRepresentation,
)

from ufre.ufre_models import (
    EvidenceFeature,
    StageEvidence,
)

from ufre.exceptions import (
    InvalidUFREInputError,
    InvalidUFREOutputError,
)


class BaseReasoner(nn.Module, ABC):
    """
    Abstract base class for all stage-wise reasoning modules.

    Every subclass processes each representation stage independently
    and produces a corresponding StageEvidence object.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        representations: RepresentationOutput,
    ) -> tuple[
        StageEvidence,
        StageEvidence,
        StageEvidence,
        StageEvidence,
    ]:
        """
        Process all representation stages.

        Parameters
        ----------
        representations
            Refined representation output.

        Returns
        -------
        tuple
            Stage evidence for stage1-stage4.
        """

        self._validate_input(representations)

        outputs = []

        for stage in representations.stages:

            evidence = self.reason_stage(stage)

            self._validate_stage_output(evidence)

            outputs.append(evidence)

        return tuple(outputs)
    @abstractmethod
    def reason_stage(
        self,
        stage: StageRepresentation,
    ) -> StageEvidence:
        """
        Process a single representation stage.

        Parameters
        ----------
        stage
            A StageRepresentation object from RepresentationOutput.

        Returns
        -------
        StageEvidence
            Forensic evidence extracted from the stage.
        """
        raise NotImplementedError

    def _validate_input(
        self,
        representations: RepresentationOutput,
    ) -> None:
        """
        Validate the input RepresentationOutput.

        Raises
        ------
        InvalidUFREInputError
            If the input is invalid.
        """

        if not isinstance(representations, RepresentationOutput):
            raise InvalidUFREInputError(
                f"Expected RepresentationOutput, got "
                f"{type(representations).__name__}."
            )

        try:
            representations.validate()
        except Exception as exc:
            raise InvalidUFREInputError(
                f"Invalid RepresentationOutput: {exc}"
            ) from exc

    def _validate_stage_output(
        self,
        evidence: StageEvidence,
    ) -> None:
        """
        Validate a StageEvidence produced by a subclass.

        Raises
        ------
        InvalidUFREOutputError
            If the produced evidence is invalid.
        """

        if not isinstance(evidence, StageEvidence):
            raise InvalidUFREOutputError(
                f"Expected StageEvidence, got "
                f"{type(evidence).__name__}."
            )

        try:
            evidence.validate()
        except Exception as exc:
            raise InvalidUFREOutputError(
                f"Invalid StageEvidence: {exc}"
            ) from exc
    @staticmethod
    def clone_evidence(
        evidence: EvidenceFeature,
    ) -> EvidenceFeature:
        """
        Create a deep copy of an EvidenceFeature.

        Parameters
        ----------
        evidence
            Evidence to clone.

        Returns
        -------
        EvidenceFeature
            Independent copy.
        """

        evidence.validate()

        return EvidenceFeature(
            stage=evidence.stage,
            evidence_type=evidence.evidence_type,
            tensor=evidence.tensor.clone(),
            confidence=evidence.confidence,
            metadata=dict(evidence.metadata),
        )

    @staticmethod
    def check_same_shape(
        tensor1: Tensor,
        tensor2: Tensor,
    ) -> None:
        """
        Ensure tensors have identical shapes.
        """

        if tensor1.shape != tensor2.shape:
            raise ValueError(
                f"Shape mismatch: "
                f"{tensor1.shape} != {tensor2.shape}"
            )

    @staticmethod
    def check_same_device(
        tensor1: Tensor,
        tensor2: Tensor,
    ) -> None:
        """
        Ensure tensors are on the same device.
        """

        if tensor1.device != tensor2.device:
            raise ValueError(
                f"Device mismatch: "
                f"{tensor1.device} != {tensor2.device}"
            )

    @staticmethod
    def check_same_dtype(
        tensor1: Tensor,
        tensor2: Tensor,
    ) -> None:
        """
        Ensure tensors share the same dtype.
        """

        if tensor1.dtype != tensor2.dtype:
            raise ValueError(
                f"Dtype mismatch: "
                f"{tensor1.dtype} != {tensor2.dtype}"
            )

    @staticmethod
    def ensure_finite(
        tensor: Tensor,
    ) -> None:
        """
        Ensure tensor contains only finite values.
        """

        if not torch.isfinite(tensor).all():
            raise ValueError(
                "Tensor contains NaN or Inf values."
            )