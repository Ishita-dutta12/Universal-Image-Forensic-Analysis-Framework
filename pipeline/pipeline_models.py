"""
UIFAF Pipeline Data Models

Defines the shared models used by the Pipeline module.

Author: Ishita Dutta
Project: UIFAF
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from preprocessing.image_info import ImageInfo
from preprocessing.decoded_image import DecodedImage

from metadata.metadata_models import MetadataInfo

from backbone.backbone_models import BackboneOutput

from attention.attention_models import AttentionOutput

from representation.representation_models import (
    RepresentationOutput,
)

from ufre.ufre_models import UFREOutput

from visualization.localization_models import (
    LocalizationResult,
)

from report_generator.report_models import (
    ForensicReport,
)


# ============================================================
# Pipeline Request
# ============================================================

@dataclass(slots=True)
class PipelineRequest:
    """
    Input request to the UIFAF pipeline.
    """

    image_path: Path


# ============================================================
# Pipeline Context
# ============================================================

@dataclass(slots=True)
class PipelineContext:
    """
    Stores outputs from every pipeline stage.

    Each module writes exactly one field.
    """

    # -------------------------
    # Preprocessing
    # -------------------------

    image_info: ImageInfo | None = None

    decoded_image: DecodedImage | None = None

    metadata: MetadataInfo | None = None

    # -------------------------
    # Feature Extraction
    # -------------------------
    backbone_input: torch.Tensor | None = None
    backbone: BackboneOutput | None = None

    attention: AttentionOutput | None = None

    representation: RepresentationOutput | None = None

    cross_representation: RepresentationOutput | None = None

    # -------------------------
    # Reasoning
    # -------------------------

    reasoning: UFREOutput | None = None

    # -------------------------
    # Explainability
    # -------------------------
    backbone_input: torch.Tensor | None = None
    localization: LocalizationResult | None = None


# ============================================================
# Pipeline Statistics
# ============================================================

@dataclass(slots=True)
class PipelineStatistics:
    """
    Runtime information collected during execution.
    """

    total_modules: int = 9

    completed_modules: int = 0

    failed_module: str | None = None

    execution_times: dict[str, float] = field(
        default_factory=dict
    )

    additional_info: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# Pipeline Result
# ============================================================

@dataclass(slots=True)
class PipelineResult:
    """
    Final object returned by the pipeline.
    """

    report: ForensicReport

    context: PipelineContext

    statistics: PipelineStatistics

    success: bool = True

    message: str = "Pipeline executed successfully."