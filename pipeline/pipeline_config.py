"""
UIFAF Pipeline Configuration.

Central configuration used by the UIFAF pipeline.

Author: Ishita Dutta
Project: UIFAF
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(slots=True)
class PipelineConfig:
    """
    Configuration for the UIFAF pipeline.
    """

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device: torch.device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # --------------------------------------------------
    # Model Checkpoints
    # --------------------------------------------------

    backbone_checkpoint: Path | None = None

    attention_checkpoint: Path | None = None

    representation_checkpoint: Path | None = None

    reasoning_checkpoint: Path | None = None

    # --------------------------------------------------
    # Output
    # --------------------------------------------------

    output_directory: Path = Path("outputs")

    save_heatmap: bool = True

    save_overlay: bool = True

    save_json_report: bool = True

    # --------------------------------------------------
    # Runtime
    # --------------------------------------------------

    batch_size: int = 1

    use_amp: bool = False

    deterministic: bool = True

    num_workers: int = 0

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    def validate(self) -> None:
        """
        Validate configuration.
        """

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        if self.num_workers < 0:
            raise ValueError(
                "num_workers cannot be negative."
            )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )