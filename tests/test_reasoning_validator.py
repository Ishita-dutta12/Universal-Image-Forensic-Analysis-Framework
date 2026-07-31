import pytest
import torch

from representation.representation_models import (
    RepresentationFeature,
    RepresentationOutput,
    StageRepresentation,
)

from ufre.reasoning_validator import ReasoningValidator

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

def make_tensor(shape=(1, 8, 16, 16)):
    return torch.randn(*shape)


def make_confidence():
    return ConfidenceEstimate(
        confidence=0.9,
        uncertainty=0.1,
    )


def make_representation_feature(stage="stage1"):
    return RepresentationFeature(
        stage=stage,
        tensor=make_tensor(),
        representation_type="spatial",
    )


def make_stage_representation(stage="stage1"):
    return StageRepresentation(
        stage=stage,
        spatial=make_representation_feature(stage),
    )


def make_representation_output():
    return RepresentationOutput(
        stage1=make_stage_representation("stage1"),
        stage2=make_stage_representation("stage2"),
        stage3=make_stage_representation("stage3"),
        stage4=make_stage_representation("stage4"),
        original_size=(256, 256),
    )


def make_evidence_feature(stage="stage1"):
    return EvidenceFeature(
        stage=stage,
        evidence_type=EvidenceType.NOISE,
        tensor=make_tensor(),
        confidence=make_confidence(),
    )


def make_stage_evidence(stage="stage1"):
    return StageEvidence(
        stage=stage,
        evidences=(make_evidence_feature(stage),),
        fused_tensor=make_tensor(),
        confidence=make_confidence(),
    )


# ==========================================================
# validate_representation_output
# ==========================================================

class TestRepresentationValidation:

    def test_valid_representation(self):
        rep = make_representation_output()
        ReasoningValidator.validate_representation_output(rep)

    def test_invalid_type(self):
        with pytest.raises(InvalidUFREInputError):
            ReasoningValidator.validate_representation_output("abc")


# ==========================================================
# validate_stage_evidence
# ==========================================================

class TestStageEvidenceValidation:

    def test_valid_stage(self):
        stage = make_stage_evidence()
        ReasoningValidator.validate_stage_evidence(stage)

    def test_invalid_stage_type(self):
        with pytest.raises(InvalidUFREOutputError):
            ReasoningValidator.validate_stage_evidence("abc")


# ==========================================================
# validate_stage_collection
# ==========================================================

class TestStageCollectionValidation:

    def test_valid_collection(self):
        stages = (
            make_stage_evidence("stage1"),
            make_stage_evidence("stage2"),
            make_stage_evidence("stage3"),
            make_stage_evidence("stage4"),
        )

        ReasoningValidator.validate_stage_collection(stages)

    def test_not_tuple(self):
        with pytest.raises(InvalidUFREOutputError):
            ReasoningValidator.validate_stage_collection([])

    def test_wrong_number_of_stages(self):
        stages = (
            make_stage_evidence("stage1"),
            make_stage_evidence("stage2"),
        )

        with pytest.raises(InvalidUFREOutputError):
            ReasoningValidator.validate_stage_collection(stages)

    def test_invalid_stage_inside_tuple(self):
        stages = (
            make_stage_evidence("stage1"),
            make_stage_evidence("stage2"),
            "bad",
            make_stage_evidence("stage4"),
        )

        with pytest.raises(InvalidUFREOutputError):
            ReasoningValidator.validate_stage_collection(stages)
    # ==========================================================
# validate_evidence_feature
# ==========================================================

class TestEvidenceFeatureValidation:

    def test_valid_evidence(self):
        evidence = make_evidence_feature()
        ReasoningValidator.validate_evidence_feature(evidence)

    def test_invalid_evidence_type(self):
        with pytest.raises(InvalidUFREOutputError):
            ReasoningValidator.validate_evidence_feature("bad")


# ==========================================================
# validate_confidence
# ==========================================================

class TestConfidenceValidation:

    def test_valid_confidence(self):
        confidence = make_confidence()
        ReasoningValidator.validate_confidence(confidence)

    def test_invalid_confidence_type(self):
        with pytest.raises(InvalidUFREOutputError):
            ReasoningValidator.validate_confidence(0.9)


# ==========================================================
# validate_same_shape
# ==========================================================

class TestSameShapeValidation:

    def test_same_shape_success(self):
        tensor1 = make_tensor((1, 8, 16, 16))
        tensor2 = make_tensor((1, 8, 16, 16))

        ReasoningValidator.validate_same_shape(
            tensor1,
            tensor2,
        )

    def test_shape_mismatch(self):
        tensor1 = make_tensor((1, 8, 16, 16))
        tensor2 = make_tensor((1, 4, 16, 16))

        with pytest.raises(InvalidUFREInputError):
            ReasoningValidator.validate_same_shape(
                tensor1,
                tensor2,
            )


# ==========================================================
# validate_same_device
# ==========================================================

class TestSameDeviceValidation:

    def test_same_device_success(self):
        tensor1 = make_tensor()
        tensor2 = make_tensor()

        ReasoningValidator.validate_same_device(
            tensor1,
            tensor2,
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available",
    )
    def test_device_mismatch(self):
        cpu = make_tensor()
        gpu = make_tensor().cuda()

        with pytest.raises(InvalidUFREInputError):
            ReasoningValidator.validate_same_device(
                cpu,
                gpu,
            )


# ==========================================================
# validate_same_dtype
# ==========================================================

class TestSameDtypeValidation:

    def test_same_dtype_success(self):
        tensor1 = torch.randn(2, 2)
        tensor2 = torch.randn(2, 2)

        ReasoningValidator.validate_same_dtype(
            tensor1,
            tensor2,
        )

    def test_dtype_mismatch(self):
        tensor1 = torch.randn(
            2,
            2,
            dtype=torch.float32,
        )

        tensor2 = torch.randn(
            2,
            2,
            dtype=torch.float64,
        )

        with pytest.raises(InvalidUFREInputError):
            ReasoningValidator.validate_same_dtype(
                tensor1,
                tensor2,
            )


# ==========================================================
# validate_finite
# ==========================================================

class TestFiniteValidation:

    def test_finite_tensor(self):
        tensor = torch.randn(3, 3)

        ReasoningValidator.validate_finite(
            tensor,
        )

    def test_nan_tensor(self):
        tensor = torch.tensor(
            [[1.0, float("nan")]]
        )

        with pytest.raises(InvalidUFREInputError):
            ReasoningValidator.validate_finite(
                tensor,
            )

    def test_inf_tensor(self):
        tensor = torch.tensor(
            [[1.0, float("inf")]]
        )

        with pytest.raises(InvalidUFREInputError):
            ReasoningValidator.validate_finite(
                tensor,
            )

    def test_invalid_tensor_type(self):
        with pytest.raises(InvalidUFREInputError):
            ReasoningValidator.validate_finite(
                "tensor"
            )