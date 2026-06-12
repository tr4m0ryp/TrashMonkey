"""Auto-boxing chain for classification-only datasets (T3).

Primary: Grounding DINO (text prompt localizes; class ID comes from the
source mapping). Fallback: BiRefNet mask -> enclosing rect. Last resort:
center box, always flagged for review.
"""

from trashmonkey.data.autobox.birefnet import BIREFNET_SESSION, build_birefnet_backend
from trashmonkey.data.autobox.chain import (
    IMAGE_SUFFIXES,
    PROVENANCE_FILENAME,
    box_directory,
)
from trashmonkey.data.autobox.dino import build_dino_backend
from trashmonkey.data.autobox.geometry import (
    center_box,
    clamp_box,
    largest_component_box,
    mask_to_box,
    yolo_line,
)
from trashmonkey.data.autobox.types import (
    CENTER_BOX_MARGIN,
    DEFAULT_TEXT_THRESHOLD,
    MASK_MAX_AREA_FRAC,
    MASK_MIN_AREA_FRAC,
    MIN_BOX_CONFIDENCE,
    PROMPTS,
    XYXY,
    BoxRecord,
    Detection,
    DinoPredictFn,
    MaskFn,
    Method,
    ProgressFn,
)

__all__ = [
    "BIREFNET_SESSION",
    "CENTER_BOX_MARGIN",
    "DEFAULT_TEXT_THRESHOLD",
    "IMAGE_SUFFIXES",
    "MASK_MAX_AREA_FRAC",
    "MASK_MIN_AREA_FRAC",
    "MIN_BOX_CONFIDENCE",
    "PROMPTS",
    "PROVENANCE_FILENAME",
    "XYXY",
    "BoxRecord",
    "Detection",
    "DinoPredictFn",
    "MaskFn",
    "Method",
    "ProgressFn",
    "box_directory",
    "build_birefnet_backend",
    "build_dino_backend",
    "center_box",
    "clamp_box",
    "largest_component_box",
    "mask_to_box",
    "yolo_line",
]
