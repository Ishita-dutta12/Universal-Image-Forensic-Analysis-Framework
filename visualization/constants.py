"""
UIFAF Visualization Constants

Central configuration values used throughout the
Visualization & Localization module.
"""

from __future__ import annotations

# ============================================================
# Explainability Methods
# ============================================================

SUPPORTED_EXPLAINERS = (
    "gradcam",
    "gradcam++",
    "scorecam",
    "eigencam",
    "layercam",
)

DEFAULT_EXPLAINER = "gradcam"

# ============================================================
# Grad-CAM
# ============================================================

# If not explicitly specified, use the final convolution block.
DEFAULT_TARGET_LAYER = "backbone.features"

# Normalize CAM values to [0, 1]
NORMALIZE_CAM = True

# ============================================================
# Heatmap
# ============================================================

HEATMAP_COLORMAP = "jet"

HEATMAP_ALPHA = 0.45

HEATMAP_DPI = 300

# ============================================================
# Overlay
# ============================================================

OVERLAY_ALPHA = 0.45

OVERLAY_FORMAT = "png"

# ============================================================
# Localization
# ============================================================

LOCALIZATION_THRESHOLD = 0.50

MIN_REGION_AREA = 100

MAX_REGIONS = 10

# ============================================================
# Output
# ============================================================

OUTPUT_DIRECTORY = "outputs"

HEATMAP_DIRECTORY = "outputs/heatmaps"

OVERLAY_DIRECTORY = "outputs/overlays"

MASK_DIRECTORY = "outputs/masks"

# ============================================================
# Image
# ============================================================

SUPPORTED_IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
)

# ============================================================
# Numerical Stability
# ============================================================

EPSILON = 1e-8