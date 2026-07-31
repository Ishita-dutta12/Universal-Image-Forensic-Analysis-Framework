"""
Unit tests for base_reasoner.py

Tests the abstract BaseReasoner class.

Project : UIFAF
"""

from __future__ import annotations

import pytest
import torch

from representation.representation_models import (
    RepresentationFeature,
    RepresentationOutput,
    StageRepresentation,
)


from ufre.base_reasoner import BaseReasoner

from ufre.ufre_models import (
    ConfidenceEstimate,
    EvidenceFeature,
    EvidenceType,
    StageEvidence,
)

from ufre.exceptions import (
    InvalidUFREInputError,
    InvalidUFREOutputError,
)


# ==========================================================
# Helper Builders
# ==========================================================


def make_tensor(
    shape=(2, 32, 16, 16),
):
    return torch.randn(*shape)


def make_confidence():
    return ConfidenceEstimate(
        confidence=0.9,
        uncertainty=0.1,
    )


def make_representation_feature(
    stage="stage1",
    representation_type="spatial",
):
    return RepresentationFeature(
        stage=stage,
        tensor=make_tensor(),
        representation_type=representation_type,
    )


def make_stage_representation(
    stage="stage1",
):
    return StageRepresentation(
        stage=stage,
        spatial=make_representation_feature(
            stage,
            "spatial",
        ),
    )


def make_representation_output():

    return RepresentationOutput(

        stage1=make_stage_representation("stage1"),

        stage2=make_stage_representation("stage2"),

        stage3=make_stage_representation("stage3"),

        stage4=make_stage_representation("stage4"),

        original_size=(512, 512),
    )


def make_evidence_feature(
    stage="stage1",
):
    return EvidenceFeature(
        stage=stage,
        evidence_type=EvidenceType.NOISE,
        tensor=make_tensor(),
        confidence=make_confidence(),
        metadata={},
    )


def make_stage_evidence(
    stage="stage1",
):
    evidence = make_evidence_feature(stage)

    return StageEvidence(
        stage=stage,
        evidences=(evidence,),
        fused_tensor=make_tensor(),
        confidence=make_confidence(),
        metadata={},
    )


# ==========================================================
# Dummy Implementation
# ==========================================================


class DummyReasoner(BaseReasoner):

    def reason_stage(
        self,
        stage,
    ):
        return make_stage_evidence(stage.stage)


class InvalidReasoner(BaseReasoner):

    def reason_stage(
        self,
        stage,
    ):
        return "invalid output"


# ==========================================================
# Constructor
# ==========================================================


class TestConstructor:

    def test_create_reasoner(self):

        model = DummyReasoner()

        assert isinstance(
            model,
            BaseReasoner,
        )


# ==========================================================
# Forward
# ==========================================================


class TestForward:

    def test_forward_returns_four_stage_outputs(self):

        model = DummyReasoner()

        representations = make_representation_output()

        outputs = model(representations)

        assert isinstance(outputs, tuple)

        assert len(outputs) == 4

        for output in outputs:

            assert isinstance(
                output,
                StageEvidence,
            )

    def test_stage_order_preserved(self):

        model = DummyReasoner()

        outputs = model(
            make_representation_output()
        )

        assert outputs[0].stage == "stage1"

        assert outputs[1].stage == "stage2"

        assert outputs[2].stage == "stage3"

        assert outputs[3].stage == "stage4"

    def test_invalid_input_type(self):

        model = DummyReasoner()

        with pytest.raises(
            InvalidUFREInputError
        ):

            model("invalid")

    def test_invalid_reasoner_output(self):

        model = InvalidReasoner()

        with pytest.raises(
            InvalidUFREOutputError
        ):

            model(
                make_representation_output()
            )
    # ==========================================================
# _validate_input
# ==========================================================


class TestValidateInput:

    def test_validate_input_success(self):

        model = DummyReasoner()

        representations = make_representation_output()

        model._validate_input(representations)

    def test_validate_input_invalid_type(self):

        model = DummyReasoner()

        with pytest.raises(
            InvalidUFREInputError
        ):
            model._validate_input(123)

    def test_validate_input_invalid_representation(self):

        model = DummyReasoner()

        rep = make_representation_output()

        object.__setattr__(
            rep,
            "original_size",
            (-1, 100),
        )

        with pytest.raises(
            InvalidUFREInputError
        ):
            model._validate_input(rep)


# ==========================================================
# _validate_stage_output
# ==========================================================


class TestValidateStageOutput:

    def test_validate_stage_output_success(self):

        model = DummyReasoner()

        evidence = make_stage_evidence()

        

        model._validate_stage_output(evidence)

    def test_validate_stage_output_invalid_type(self):

        model = DummyReasoner()

        with pytest.raises(
            InvalidUFREOutputError
        ):
            model._validate_stage_output(
                "invalid"
            )

    def test_validate_stage_output_invalid_stage(self):

        model = DummyReasoner()

        invalid_confidence = ConfidenceEstimate(
        confidence=2.0,
        uncertainty=0.0,
        )

        evidence = StageEvidence(
            stage="stage1",
            evidences=(make_evidence_feature(),),
            fused_tensor=make_tensor(),
            confidence=invalid_confidence,
            metadata={},
        )

        with pytest.raises(InvalidUFREOutputError):
            model._validate_stage_output(evidence)


# ==========================================================
# clone_evidence
# ==========================================================


class TestCloneEvidence:

    def test_clone_returns_new_instance(self):

        evidence = make_evidence_feature()

        cloned = BaseReasoner.clone_evidence(
            evidence
        )

        assert cloned is not evidence

    def test_clone_tensor_independent(self):

        evidence = make_evidence_feature()

        cloned = BaseReasoner.clone_evidence(
            evidence
        )

        cloned.tensor.add_(100)

        assert not torch.equal(
            evidence.tensor,
            cloned.tensor,
        )

    def test_clone_metadata_independent(self):

        evidence = make_evidence_feature()

        evidence.metadata["name"] = "noise"

        cloned = BaseReasoner.clone_evidence(
            evidence
        )

        cloned.metadata["name"] = "edited"

        assert (
            evidence.metadata["name"]
            == "noise"
        )

        assert (
            cloned.metadata["name"]
            == "edited"
        )

    def test_clone_preserves_stage(self):

        evidence = make_evidence_feature()

        cloned = BaseReasoner.clone_evidence(
            evidence
        )

        assert (
            cloned.stage
            == evidence.stage
        )

    def test_clone_preserves_type(self):

        evidence = make_evidence_feature()

        cloned = BaseReasoner.clone_evidence(
            evidence
        )

        assert (
            cloned.evidence_type
            == evidence.evidence_type
        )

    def test_clone_preserves_confidence(self):

        evidence = make_evidence_feature()

        cloned = BaseReasoner.clone_evidence(
            evidence
        )

        assert (
            cloned.confidence.confidence
            == evidence.confidence.confidence
        )

        assert (
            cloned.confidence.uncertainty
            == evidence.confidence.uncertainty
        )


# ==========================================================
# Tensor Utility Functions
# ==========================================================


class TestTensorUtilities:

    def test_same_shape_success(self):

        a = torch.randn(2, 3)

        b = torch.randn(2, 3)

        BaseReasoner.check_same_shape(
            a,
            b,
        )

    def test_same_shape_failure(self):

        a = torch.randn(2, 3)

        b = torch.randn(5, 3)

        with pytest.raises(ValueError):

            BaseReasoner.check_same_shape(
                a,
                b,
            )

    def test_same_device_success(self):

        a = torch.randn(2, 3)

        b = torch.randn(2, 3)

        BaseReasoner.check_same_device(
            a,
            b,
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA unavailable",
    )
    def test_same_device_failure(self):

        cpu = torch.randn(2, 3)

        gpu = torch.randn(
            2,
            3,
            device="cuda",
        )

        with pytest.raises(ValueError):

            BaseReasoner.check_same_device(
                cpu,
                gpu,
            )

    def test_same_dtype_success(self):

        a = torch.randn(
            2,
            dtype=torch.float32,
        )

        b = torch.randn(
            2,
            dtype=torch.float32,
        )

        BaseReasoner.check_same_dtype(
            a,
            b,
        )

    def test_same_dtype_failure(self):

        a = torch.randn(
            2,
            dtype=torch.float32,
        )

        b = torch.randn(
            2,
            dtype=torch.float64,
        )

        with pytest.raises(ValueError):

            BaseReasoner.check_same_dtype(
                a,
                b,
            )

    def test_ensure_finite_success(self):

        tensor = torch.randn(4, 4)

        BaseReasoner.ensure_finite(
            tensor
        )

    def test_ensure_finite_nan(self):

        tensor = torch.randn(4, 4)

        tensor[0, 0] = float("nan")

        with pytest.raises(ValueError):

            BaseReasoner.ensure_finite(
                tensor
            )

    def test_ensure_finite_inf(self):

        tensor = torch.randn(4, 4)

        tensor[0, 0] = float("inf")

        with pytest.raises(ValueError):

            BaseReasoner.ensure_finite(
                tensor
            )
    def test_clone_invalid_evidence(self):

        evidence = make_evidence_feature()

        evidence.tensor[0, 0, 0, 0] = float("nan")

        with pytest.raises(ValueError):
            BaseReasoner.clone_evidence(evidence)