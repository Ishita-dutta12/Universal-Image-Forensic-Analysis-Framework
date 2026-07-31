"""
Decision Engine

Author: Ishita Dutta
Project: UIFAF

Description
-----------
Final forensic decision module.

This module receives outputs from the complete pipeline
(UFRE, Metadata, Localization) and produces the final
DecisionOutput used by the Report Generator.

Responsibilities
----------------
1. Predict image class
2. Fuse metadata evidence
3. Fuse localization evidence
4. Compute confidence
5. Compute risk
6. Generate findings
7. Produce DecisionOutput
"""

from __future__ import annotations

from importlib import metadata
from typing import List

from matplotlib.pylab import indices
from tensorboard import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
import metadata

from pipeline.pipeline_models import (
    PipelineContext,
    PipelineStatistics,
)

from metadata.metadata_models import MetadataInfo


from visualization.localization_models import (
    LocalizationResult,
)

from ufre.ufre_models import (
    UFREOutput,
)

from report_generator.report_models import (
    FinalVerdict,
)

from .decision_models import (
    ImageClass,
    ClassificationResult,
    DecisionOutput,
    DecisionStatistics,
    EvidenceScore,
    FindingResult,
)

from .exceptions import (
    InvalidDecisionInputError,
    ClassificationError,
)
# ============================================================
# Network Configuration
# ============================================================

_EMBEDDING_DIM = 512

_HIDDEN_DIM = 256

_DROPOUT = 0.20


# ============================================================
# Decision Thresholds
# ============================================================

_HIGH_CONFIDENCE = 0.85

_MEDIUM_CONFIDENCE = 0.60

_LOW_CONFIDENCE = 0.35


_HIGH_RISK = 0.75

_MEDIUM_RISK = 0.45


_METADATA_BONUS = 0.08

_LOCALIZATION_BONUS = 0.05


_EPS = 1e-8

class DecisionEngine(nn.Module):
    """
    Final decision module of UIFAF.

    Inputs
    ------
    UFREOutput

    MetadataInfo

    LocalizationResult

    Outputs
    -------
    DecisionOutput
    """

    def __init__(self):

        super().__init__()
        
        self.classifier = nn.Sequential(

            nn.Linear(
                _EMBEDDING_DIM,
                _HIDDEN_DIM,
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(
                _DROPOUT,
            ),

            nn.Linear(
                _HIDDEN_DIM,
                len(ImageClass),
            ),
        )
        self.softmax = nn.Softmax(dim=1)
    def forward(
    self,
    reasoning: UFREOutput,
    metadata: MetadataInfo | None = None,
    localization: LocalizationResult | None = None,
) -> DecisionOutput:
    

    # -----------------------------------------------------
    # Validate Inputs
    # -----------------------------------------------------

        if reasoning is None:
            raise InvalidDecisionInputError(
            "UFREOutput cannot be None."
        )

    # -----------------------------------------------------
    # Step 1
    # Extract forensic embedding
    # -----------------------------------------------------

        embedding = self._extract_embedding(
            reasoning,
    )

    # -----------------------------------------------------
    # Step 2
    # Neural classification
    # -----------------------------------------------------

        probabilities = self._predict_classes(
            embedding,
    )

    # -----------------------------------------------------
    # Step 3
    # Metadata refinement
    # -----------------------------------------------------

        probabilities = self._adjust_metadata(
            probabilities,
        metadata,
    )

    # -----------------------------------------------------
    # Step 4
    # Localization refinement
    # -----------------------------------------------------

        probabilities = self._adjust_localization(
            probabilities,
        localization,
    )

    # -----------------------------------------------------
    # Step 5
    # Convert probabilities into evidence scores
    # -----------------------------------------------------

        evidence_scores = self._build_evidence_scores(
            probabilities,
    )

    # -----------------------------------------------------
    # Step 6
    # Final classification
    # -----------------------------------------------------

        classification = self._classify(
            probabilities,
    )

    # -----------------------------------------------------
    # Step 7
    # Confidence estimation
    # -----------------------------------------------------

        confidence = self._compute_confidence(
            reasoning,
    )

    # -----------------------------------------------------
    # Step 8
    # Risk estimation
    # -----------------------------------------------------

        risk = self._compute_risk(
            probabilities,
        confidence,
    )

    # -----------------------------------------------------
    # Step 9
    # Final verdict
    # -----------------------------------------------------

        verdict = self._determine_verdict(
            classification,
            confidence,
            risk,
    )

    # -----------------------------------------------------
    # Step 10
    # Generate findings
    # -----------------------------------------------------

        findings = self._generate_findings(
            reasoning,
            metadata,
            localization,
            evidence_scores,
    )

    # -----------------------------------------------------
    # Step 11
    # Statistics
    # -----------------------------------------------------

        statistics = self._build_statistics(
            reasoning,
            confidence,
    )

    # -----------------------------------------------------
    # Step 12
    # Build output
    # -----------------------------------------------------

        output = DecisionOutput(
            classification=classification,
            final_verdict=verdict,
            risk_score=risk,
            confidence=confidence,
            uncertainty=max(
                0.0,
                1.0 - confidence,
            ),
            evidence_scores=evidence_scores,
            findings=findings,
            statistics=statistics,
        )

        # Validate the generated output
        self._validate_output(output)

        return output

    def _extract_embedding(
    self,
    reasoning: UFREOutput,
) -> torch.Tensor:

        embedding = reasoning.forensic_embedding

        if embedding is None:
            raise ClassificationError(
            "UFRE returned no forensic embedding."
        )

        if not isinstance(embedding, torch.Tensor):
            raise ClassificationError(
            "Forensic embedding must be torch.Tensor."
        )

        if embedding.ndim != 2:
            raise ClassificationError(
            "Forensic embedding must be 2-dimensional."
        )

        if embedding.shape[1] != _EMBEDDING_DIM:
            raise ClassificationError(
            f"Expected embedding size {_EMBEDDING_DIM}, "
            f"got {embedding.shape[1]}."
        )

        if not torch.isfinite(embedding).all():
            raise ClassificationError(
            "Embedding contains NaN or Inf."
        )

        return embedding
    
    def _predict_classes(
    self,
    embedding: torch.Tensor,
) -> torch.Tensor:
    

        logits = self.classifier(
        embedding,
    )

        probabilities = self.softmax(
        logits,
    )

        if not torch.isfinite(probabilities).all():
            raise ClassificationError(
            "Classifier produced invalid probabilities."
        )

        return probabilities
    
    def _classify(
    self,
    probabilities: torch.Tensor,
) -> ClassificationResult:

        probs = probabilities.squeeze(0)

        values, indices = torch.topk(
        probs,
        k=2,
    )

        classes = list(ImageClass)

        primary = classes[indices[0].item()]

        secondary = None
        if len(indices) > 1:
            secondary = classes[indices[1].item()]

        confidence = float(values[0].item())

        return ClassificationResult(
        primary_class=primary,
        secondary_class=secondary,
        probability=confidence,
    )
        
    def _build_evidence_scores(
    self,
    probabilities: torch.Tensor,
) -> list[EvidenceScore]:
    

        probs = probabilities.squeeze(0)

        

        scores = []

        classes = list(ImageClass)

        for image_class, probability in zip(classes, probs):

            scores.append(

                EvidenceScore(

                image_class=image_class,

                probability=float(
                    probability.item()
                ),
            )
        )

        return scores 
    def _adjust_metadata(
    self,
    probabilities: torch.Tensor,
    metadata: MetadataInfo | None,
) -> torch.Tensor:

        if metadata is None:
            return probabilities

        probabilities = probabilities.clone()

    # --------------------------------------------------
    # GPS removed / missing
    # --------------------------------------------------

        if metadata.gps is None:

            index = list(ImageClass).index(
                ImageClass.METADATA_TAMPERING
        )

            probabilities[:, index] += (
            _METADATA_BONUS / 2
        )

        probabilities = probabilities / (
            probabilities.sum(
            dim=1,
            keepdim=True,
        )
        + _EPS
    )

        return probabilities   
    
    def _adjust_localization(
    self,
    probabilities: torch.Tensor,
    localization: LocalizationResult | None,
) -> torch.Tensor:
   

        if localization is None:
            return probabilities

        probabilities = probabilities.clone()

        if not localization.bounding_boxes:
            return probabilities

        confidence = localization.confidence

        if confidence < 0.50:
            return probabilities

        classes = list(ImageClass)

        suspicious = [

            ImageClass.AI_EDITED,

        ImageClass.COPY_MOVE,

        ImageClass.SPLICING,

        ImageClass.OBJECT_REMOVAL,
    ]

        for image_class in suspicious:

            idx = classes.index(image_class)

            probabilities[:, idx] += (

            _LOCALIZATION_BONUS
            * confidence

        )

        probabilities = probabilities / (

            probabilities.sum(
            dim=1,
            keepdim=True,
        )
        + _EPS
    )

        return probabilities
    
    def _compute_confidence(
    self,
    reasoning: UFREOutput,
) -> float:
    

        estimate = reasoning.global_confidence

        if estimate is None:
            raise ClassificationError(
            "Missing global confidence."
        )

        confidence = float(
        estimate.confidence
    )

        confidence = max(
        0.0,
        min(
            confidence,
            1.0,
        ),
    )

        return confidence
    def _compute_risk(
    self,
    probabilities: torch.Tensor,
    confidence: float,
) -> float:
    

        probs = probabilities.squeeze(0)

        classes = list(ImageClass)

        authentic_idx = list(ImageClass).index(
        ImageClass.AUTHENTIC
    )

        authentic_probability = float(
            probs[authentic_idx].item()
    )

        manipulation_probability = (
        1.0 - authentic_probability
    )

        risk = (
        manipulation_probability
        * confidence
    )

        return max(
        0.0,
        min(
                risk,
            1.0,
        ),
    )
        
    def _determine_verdict(
    self,
    classification: ClassificationResult,
    confidence: float,
    risk: float,
) -> FinalVerdict:
    
        image_class = classification.primary_class

    # ---------------------------------------------
    # Authentic
    # ---------------------------------------------

        if image_class == ImageClass.AUTHENTIC:

            if confidence >= _HIGH_CONFIDENCE:

                return FinalVerdict.AUTHENTIC

            return FinalVerdict.SUSPICIOUS

    # ---------------------------------------------
    # Manipulated
    # ---------------------------------------------

        if risk >= _HIGH_RISK:

            return FinalVerdict.MANIPULATED

        if risk >= _MEDIUM_RISK:

            return FinalVerdict.SUSPICIOUS

        return FinalVerdict.UNKNOWN
    
    def _generate_findings(
        self,
        reasoning: UFREOutput,
        metadata: MetadataInfo | None,
        localization: LocalizationResult | None,
        evidence_scores: list[EvidenceScore],
    ) -> list[FindingResult]:
        findings: list[FindingResult] = []

        # --------------------------------------------------
        # Top classifier prediction
        # --------------------------------------------------
        top_score = max(
            evidence_scores,
            key=lambda score: score.probability,
        )
        findings.append(
            FindingResult(

    title="Reasoning Confidence",

    description="Overall forensic reasoning confidence.",

    confidence=float(
        reasoning.global_confidence.confidence
    ),

    source_module="UFRE",

    source_branch="Reasoning",

    supporting_evidence=[
        "Global Confidence",
    ],
)
        )

        # --------------------------------------------------
        # Metadata Finding
        # --------------------------------------------------
        if metadata is not None:

            if not metadata.raw_exif:

                findings.append(
            FindingResult(
                title="Metadata Missing",
                description="Image does not contain EXIF metadata.",
                confidence=0.80,
                source_module="Metadata",
                source_branch="EXIF",
                supporting_evidence=[
                    "Missing EXIF",
                ],
            )
        )

        # --------------------------------------------------
        # Localization Finding
        # --------------------------------------------------
        if localization is not None:
            try:
                if (
                    localization.bounding_boxes
                    and localization.confidence > 0.50
                ):
                    findings.append(
                        FindingResult(
                            title="Manipulated Region Detected",
                            description=(
                                "Potential manipulated region "
                                "identified."
                            ),
                            confidence=float(
                                localization.confidence
                            ),
                            source_module="Localization",
                            source_branch="Bounding Boxes",
                            supporting_evidence=[
                                "Manipulated Region",
                            ],
                        )
                    )
            except AttributeError:
                # Localization model may differ.
                pass

        # --------------------------------------------------
        # UFRE Global Confidence
        # --------------------------------------------------
        

        return findings
    
    def _build_statistics(
        self,
        reasoning: UFREOutput,
        confidence: float,
    ) -> DecisionStatistics:
    
        summary = reasoning.evidence_summary

        return DecisionStatistics(

        evidence_count=summary.total_evidence,

        average_confidence=summary.average_confidence,

        reasoning_confidence=confidence,
    )
        
    def _validate_output(
    self,
    output: DecisionOutput,
    ) -> None:
    

        if not 0.0 <= output.confidence <= 1.0:
            raise ClassificationError(
            "Invalid confidence."
        )

        if not 0.0 <= output.risk_score <= 1.0:
            raise ClassificationError(
            "Invalid risk score."
        )

        if output.classification is None:
            raise ClassificationError(
            "Missing classification."
        )

