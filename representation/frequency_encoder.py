# frequency_encoder.py
# Generated scaffold compatible with current UIFAF Module 5.

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention.attention_models import AttentionFeatureMap, AttentionOutput
from .dwt_transform import DWTTransform
from .exceptions import RepresentationInitializationError, RepresentationError
#from .representation_models import RepresentationFeature, StageRepresentation
from .base_encoder import BaseRepresentationEncoder
#from .representation_models import RepresentationFeature
from .representation_models import (
    RepresentationFeature,
    RepresentationOutput,
    StageRepresentation,
)
class FrequencyResidualBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 32):
        super().__init__()
        if channels <= 0:
            raise RepresentationInitializationError("channels must be positive.")
        groups = min(groups, channels)
        while channels % groups != 0:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)


class SEAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class BandFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(32, channels)
        while channels % groups != 0:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, ll, lh, hl, hh):
        return self.net(torch.cat([ll, lh, hl, hh], dim=1))


class FrequencyEncoder(BaseRepresentationEncoder):
    def __init__(self, stage_channels: Dict[str, int]):
        super().__init__()
        required = {
            "stage1",
            "stage2",
            "stage3",
            "stage4",
        }

        if set(stage_channels.keys()) != required:
            raise RepresentationInitializationError(
                f"Expected stages {required}, got {set(stage_channels.keys())}"
            )
        self.dwt = DWTTransform()
        self.band_blocks = nn.ModuleDict()
        self.fusions = nn.ModuleDict()
        self.attn = nn.ModuleDict()
        self.refine = nn.ModuleDict()

        for stage, ch in stage_channels.items():
            self.band_blocks[stage] = nn.ModuleDict({
                "ll": FrequencyResidualBlock(ch),
                "lh": FrequencyResidualBlock(ch),
                "hl": FrequencyResidualBlock(ch),
                "hh": FrequencyResidualBlock(ch),
            })
            self.fusions[stage] = BandFusion(ch)
            self.attn[stage] = SEAttention(ch)

            groups = min(32, ch)
            while ch % groups != 0:
                groups -= 1

            self.refine[stage] = nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.GroupNorm(groups, ch),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.GroupNorm(groups, ch),
                nn.GELU(),
            )

        self.apply(self._init)

    def _init(self, module):

        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _encode_stage(
        self,
        feature: AttentionFeatureMap,
        ) -> RepresentationFeature:
        c = self.dwt(feature.tensor)
        c.validate()
        stage = feature.stage
        b = self.band_blocks[stage]
        ll = b["ll"](c.ll)
        lh = b["lh"](c.lh)
        hl = b["hl"](c.hl)
        hh = b["hh"](c.hh)
        x = self.fusions[stage](ll, lh, hl, hh)
        x = self.attn[stage](x)
        x = self.refine[stage](x)
        x = F.interpolate(
            x,
            size=feature.tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        if x.shape != feature.tensor.shape:
            raise RepresentationError(
                f"Expected shape {feature.tensor.shape}, "
                f"got {x.shape}."
            )
        x = x + feature.tensor
        out = RepresentationFeature(stage=stage, tensor=x, representation_type="frequency")
        out.validate()
        return out

        def forward(
            self,
            attention_output: AttentionOutput,
        ) -> dict:
            attention_output.validate()

            return {
            "stage1": self._encode_stage(
                attention_output.stage1,
            ),
            "stage2": self._encode_stage(
                attention_output.stage2,
            ),
            "stage3": self._encode_stage(
                attention_output.stage3,
            ),
            "stage4": self._encode_stage(
                attention_output.stage4,
            ),
            "original_size": attention_output.original_size,
            }

    
   
        
