from __future__ import annotations

import torch
import torch.nn as nn

from backbone.backbone_models import BackboneOutput, FeatureMap
from backbone.feature_extractor import FeatureExtractor

from attention.feature_enhancer import FeatureEnhancer

from representation.spatial_encoder import SpatialEncoder
from representation.frequency_encoder import FrequencyEncoder
from representation.context_encoder import ContextEncoder
from representation.representation_fusion import RepresentationFusion
from representation.representation_models import RepresentationOutput

from ufre.forensic_reasoner import ForensicReasoner

from decision.decision_engine import DecisionEngine
class LocalizationHead(nn.Module):
    """
    Supervised pixel-level localization head.

    Input:
        Multi-scale fused UIFAF representations.

    Output:
        Manipulation probability mask:
        [B, 1, H, W]
    """

    def __init__(self):
        super().__init__()

        # Project all stages to a common channel size.
        self.stage1_proj = nn.Conv2d(
            96, 64, kernel_size=1
        )

        self.stage2_proj = nn.Conv2d(
            192, 64, kernel_size=1
        )

        self.stage3_proj = nn.Conv2d(
            384, 64, kernel_size=1
        )

        self.stage4_proj = nn.Conv2d(
            768, 64, kernel_size=1
        )

        # After concatenation:
        # 64 * 4 = 256 channels
        self.fusion = nn.Sequential(
            nn.Conv2d(
                256, 128,
                kernel_size=3,
                padding=1
            ),
            nn.GroupNorm(
                8, 128
            ),
            nn.GELU(),

            nn.Conv2d(
                128, 64,
                kernel_size=3,
                padding=1
            ),
            nn.GroupNorm(
                8, 64
            ),
            nn.GELU(),

            nn.Conv2d(
                64, 1,
                kernel_size=1
            )
        )

    def forward(
        self,
        stage1,
        stage2,
        stage3,
        stage4,
        output_size,
    ):

        s1 = self.stage1_proj(stage1)
        s2 = self.stage2_proj(stage2)
        s3 = self.stage3_proj(stage3)
        s4 = self.stage4_proj(stage4)

        # Bring every stage to the highest spatial resolution.
        target_size = s1.shape[-2:]

        s2 = nn.functional.interpolate(
            s2,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        s3 = nn.functional.interpolate(
            s3,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        s4 = nn.functional.interpolate(
            s4,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        fused = torch.cat(
            [s1, s2, s3, s4],
            dim=1,
        )

        mask_logits = self.fusion(
            fused
        )

        mask_logits = nn.functional.interpolate(
            mask_logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        return mask_logits

class UIFAFTrainingModel(nn.Module):
    """
    Training-only UIFAF model.

    Phase 1:
        ConvNeXt V2 is frozen.

    Trainable:
        EMA Attention
        Spatial Encoder
        Frequency Encoder
        Context Encoder
        Representation Fusion
        UFRE
        Decision Classifier
    """

    def __init__(
        self,
        device: str | torch.device = "cuda",
        pretrained: bool = True,
    ) -> None:

        super().__init__()

        self.device = torch.device(device)

        stage_channels = {
            "stage1": 96,
            "stage2": 192,
            "stage3": 384,
            "stage4": 768,
        }

        # --------------------------------------------------
        # Backbone
        # --------------------------------------------------

        self.backbone = FeatureExtractor(
            pretrained=pretrained,
            device=self.device,
        )

        # Phase 1: freeze ConvNeXt
        for parameter in self.backbone.backbone.parameters():
            parameter.requires_grad = False

        self.backbone.backbone.eval()

        # --------------------------------------------------
        # EMA Attention
        # --------------------------------------------------

        self.attention = FeatureEnhancer().to(self.device)

        # --------------------------------------------------
        # Representation
        # --------------------------------------------------

        self.spatial = SpatialEncoder(
            stage_channels
        ).to(self.device)

        self.frequency = FrequencyEncoder(
            stage_channels
        ).to(self.device)

        self.context = ContextEncoder(
            stage_channels
        ).to(self.device)

        self.representation_fusion = RepresentationFusion(
            stage_channels
        ).to(self.device)

        # --------------------------------------------------
        # UFRE
        # --------------------------------------------------

        self.reasoner = ForensicReasoner(
            stage_channels=stage_channels,
        ).to(self.device)

        # --------------------------------------------------
        # Decision classifier
        # --------------------------------------------------

        self.decision_engine = DecisionEngine().to(self.device)
                # --------------------------------------------------
        # Supervised Localization Head
        # --------------------------------------------------

        self.localization_head = LocalizationHead().to(
            self.device
        )

    def _build_backbone_output(
        self,
        features,
        height: int,
        width: int,
    ) -> BackboneOutput:

        channels = self.backbone.backbone.channels
        reductions = self.backbone.backbone.reductions

        feature_maps = []

        for i, feature in enumerate(features):

            feature_maps.append(
                FeatureMap(
                    stage=f"stage{i + 1}",
                    tensor=feature,
                    channels=channels[i],
                    stride=reductions[i],
                )
            )

        return BackboneOutput(
            stage1=feature_maps[0],
            stage2=feature_maps[1],
            stage3=feature_maps[2],
            stage4=feature_maps[3],
            original_size=(height, width),
        )

    def forward(
    self,
    images: torch.Tensor,
    return_localization: bool = False,
):
        """
        Forward pass for supervised classification training.

        Parameters
        ----------
        images:
            Normalized image tensor:
            [B, 3, H, W]

        Returns
        -------
        torch.Tensor
            Raw logits:
            [B, 12]
        """

        if images.ndim != 4:
            raise ValueError(
                f"Expected [B, 3, H, W], got {images.shape}"
            )

        images = images.to(self.device)

        _, _, height, width = images.shape

        # --------------------------------------------------
        # Frozen ConvNeXt
        # --------------------------------------------------

        with torch.no_grad():
            features = self.backbone.backbone(images)

        backbone_output = self._build_backbone_output(
            features,
            height,
            width,
        )

        backbone_output.validate()

        # --------------------------------------------------
        # EMA Attention
        # --------------------------------------------------

        attention_output = self.attention(
            backbone_output
        )

        # --------------------------------------------------
        # Three representations
        # --------------------------------------------------

        spatial_output = self.spatial(
            attention_output
        )

        frequency_output = self.frequency(
            attention_output
        )

        context_output = self.context(
            attention_output
        )

        # --------------------------------------------------
        # Representation fusion
        # --------------------------------------------------

        fused = self.representation_fusion(
            spatial_output,
            frequency_output,
            context_output,
        )

        representation = RepresentationOutput(
            stage1=fused["stage1"],
            stage2=fused["stage2"],
            stage3=fused["stage3"],
            stage4=fused["stage4"],
            original_size=(height, width),
        )

        representation.validate()

        # --------------------------------------------------
        # UFRE
        # --------------------------------------------------

        reasoning = self.reasoner(
            representation
        )

        # --------------------------------------------------
        # Classification
        # --------------------------------------------------

        logits = self.decision_engine.forward_logits(
            reasoning.forensic_embedding
        )

        # --------------------------------------------------
        # Supervised Localization
        # --------------------------------------------------

        localization_logits = self.localization_head(
    fused["stage1"].fused.tensor,
    fused["stage2"].fused.tensor,
    fused["stage3"].fused.tensor,
    fused["stage4"].fused.tensor,
    output_size=(height, width),
)

        if return_localization:
            return logits, localization_logits

        return logits