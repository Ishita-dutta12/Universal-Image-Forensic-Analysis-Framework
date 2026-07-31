"""
Decoded Image Container

Author: Ishita Dutta

Universal Image Forensic Analysis Framework
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image




@dataclass(slots=True)
class DecodedImage:
    """
    Container object passed through the UIFAF pipeline.

    Stores:
    - Original PIL image (for forensic preservation)
    - Standardized RGB NumPy image
    - Image information
    - Metadata (filled by later modules)
    - Validation status
    """

    # Original file path
    filepath: Path

    # Original image after EXIF orientation correction
    original_image: Image.Image

    # RGB NumPy array used by ML models
    rgb_image: np.ndarray

    # Image information
 

    # Metadata populated by later modules
    metadata: dict[str, Any] = field(default_factory=dict)

    # Validation result
    validation_passed: bool = True