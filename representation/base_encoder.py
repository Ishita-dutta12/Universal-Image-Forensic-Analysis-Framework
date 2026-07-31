"""
Base Representation Encoder.

Provides common functionality for all representation
encoders (Spatial, Frequency and Context).

Author: Ishita Dutta
Project: UIFAF
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import torch.nn as nn

from attention.attention_models import (
    AttentionFeatureMap,
    AttentionOutput,
)

from .representation_models import (
    RepresentationFeature,
)


class BaseRepresentationEncoder(
    nn.Module,
    ABC,
):
    """
    Base class for all representation encoders.
    """

    REQUIRED_STAGES = (
        "stage1",
        "stage2",
        "stage3",
        "stage4",
    )

    def __init__(self):

        super().__init__()

    @abstractmethod
    def _encode_stage(
        self,
        feature: AttentionFeatureMap,
    ) -> RepresentationFeature:
        """
        Encode a single stage.
        """
        pass

    def forward(
        self,
        attention_output: AttentionOutput,
    ) -> Dict[str, RepresentationFeature]:

        attention_output.validate()

        return {

            "stage1": self._encode_stage(
                attention_output.stage1
            ),

            "stage2": self._encode_stage(
                attention_output.stage2
            ),

            "stage3": self._encode_stage(
                attention_output.stage3
            ),

            "stage4": self._encode_stage(
                attention_output.stage4
            ),
        }