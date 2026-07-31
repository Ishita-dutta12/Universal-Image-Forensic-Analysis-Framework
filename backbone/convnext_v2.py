"""
ConvNeXt V2 Backbone Wrapper.

Loads the official ConvNeXt V2 Tiny implementation from timm and exposes
multi-scale feature maps required by UIFAF.
"""

from __future__ import annotations

from typing import List

import timm
import torch
import torch.nn as nn

from .exceptions import (
    BackboneInitializationError,
    InvalidBackboneOutputError,
)

MODEL_NAME = "convnextv2_tiny"


class ConvNeXtV2Backbone(nn.Module):
    """
    Wrapper around ConvNeXt V2 Tiny.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        try:
            self.model = timm.create_model(
                MODEL_NAME,
                pretrained=pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3),
            )

        except Exception as exc:
            raise BackboneInitializationError(
                f"Failed to initialize {MODEL_NAME}: {exc}"
            ) from exc

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through ConvNeXt V2.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, 3, H, W).

        Returns
        -------
        List[torch.Tensor]
            Four multi-scale feature maps.
        """
        outputs = self.model(x)

        if not isinstance(outputs, (list, tuple)):
            raise InvalidBackboneOutputError(
                "Backbone must return a list or tuple of feature maps."
            )

        if len(outputs) != 4:
            raise InvalidBackboneOutputError(
                "Expected four feature maps."
            )

        for i, feature in enumerate(outputs):
            if not isinstance(feature, torch.Tensor):
                raise InvalidBackboneOutputError(
                    f"Feature map {i} is not a torch.Tensor."
                )

            if feature.ndim != 4:
                raise InvalidBackboneOutputError(
                    f"Feature map {i} must be 4-dimensional."
                )

        return list(outputs)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    @property
    def feature_info(self):
        return self.model.feature_info

    @property
    def channels(self):
        return self.model.feature_info.channels()

    @property
    def reductions(self):
        return self.model.feature_info.reduction()

        #SUPPORTED_MODELS = {
        #"atto": "convnextv2_atto",
        #"femto": "convnextv2_femto",
        #"pico": "convnextv2_pico",
        #"nano": "convnextv2_nano",
        #"tiny": "convnextv2_tiny",
        #"base": "convnextv2_base",
        #"large": "convnextv2_large",
        #"huge": "convnextv2_huge",
    #}"