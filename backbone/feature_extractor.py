"""
Feature Extractor for UIFAF.

Converts a DecodedImage into multi-scale feature maps using
ConvNeXt V2 Tiny.
"""

from __future__ import annotations

from typing import Tuple

from matplotlib import image
import torch
import torch.nn as nn
from torchvision.transforms import Normalize

from preprocessing.decoded_image import DecodedImage

from .backbone_models import (
    BackboneOutput,
    FeatureMap,
)

from .convnext_v2 import ConvNeXtV2Backbone
from .exceptions import BackboneError


class FeatureExtractor(nn.Module):
    """
    UIFAF Backbone Feature Extractor.

    Input
    -----
    DecodedImage

    Output
    ------
    BackboneOutput
    """

    def __init__(
        self,
        pretrained: bool = True,
        device: str | torch.device = "cpu",
    ):
        super().__init__()

        self.device = torch.device(device)

        self.backbone = ConvNeXtV2Backbone(
            pretrained=pretrained
        ).to(self.device)

        # ImageNet normalization
        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.last_input_tensor: torch.Tensor | None = None

    def _prepare_tensor(
        self,
        decoded_image: DecodedImage,
    ) -> torch.Tensor:
        """
        Convert DecodedImage to normalized tensor.
        """

        image = decoded_image.rgb_image

        if not isinstance(image, torch.Tensor):

            image = torch.from_numpy(image.copy()).to(self.device)

        image = image.to(torch.float32)

        if image.max() > 1.0:
            image = image / 255.0

        if image.ndim != 3:
            raise BackboneError(
        "Expected image shape (H, W, C)."
    )

        if image.shape[-1] != 3:
            raise BackboneError(
        "Expected RGB image with 3 channels."
    )

        image = image.permute(2, 0, 1)

        image = self.normalize(image)

        image = image.unsqueeze(0)

        return image.to(self.device)

    @torch.no_grad()
    def forward(
        self,
        decoded_image: DecodedImage,
    ) -> BackboneOutput:
        """
        Extract multi-scale backbone features.
        """

        tensor = self._prepare_tensor(
            decoded_image
        )

        self.last_input_tensor = tensor

        outputs = self.backbone(tensor)

        channels = self.backbone.channels

        reductions = self.backbone.reductions

        feature_maps = []

        for i, feature in enumerate(outputs):

            feature_maps.append(

                FeatureMap(

                    stage=f"stage{i+1}",

                    tensor=feature,

                    channels=channels[i],

                    stride=reductions[i],
                )

            )

        result = BackboneOutput(

            stage1=feature_maps[0],

            stage2=feature_maps[1],

            stage3=feature_maps[2],

            stage4=feature_maps[3],

          original_size=(
            decoded_image.info.height,
            decoded_image.info.width,
            )
        )

        result.validate()

        return result