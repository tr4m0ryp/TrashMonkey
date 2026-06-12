"""Shared types, prompts, and constants for the auto-boxing chain (T3).

Class IDs always come from the source dataset mapping, never from the
detector: the Grounding DINO text prompt only localizes the object.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

# One localization prompt per material class. " . " separates phrases in the
# Grounding DINO text-prompt convention. Keys must match configs/config.yaml.
PROMPTS: dict[str, str] = {
    "plastic": "plastic bottle . plastic container . plastic bag . plastic cup",
    "paper": "sheet of paper . crumpled paper . paper cup . newspaper",
    "cardboard": "cardboard box . cardboard sheet . carton . cardboard packaging",
    "metal": "metal can . aluminum can . tin can . metal container",
    "glass": "glass bottle . glass jar . glass cup",
    "organic": "food waste . fruit . vegetable . food scraps . peel",
}

# Accept a DINO box only at or above this confidence (T3: threshold ~0.25-0.35).
MIN_BOX_CONFIDENCE: float = 0.30
# Grounding DINO text threshold (phrase-token match), package default.
DEFAULT_TEXT_THRESHOLD: float = 0.25
# Reject a BiRefNet mask whose largest component covers <5% or >95% of the image.
MASK_MIN_AREA_FRAC: float = 0.05
MASK_MAX_AREA_FRAC: float = 0.95
# Last-resort center box: image minus this margin on all sides.
CENTER_BOX_MARGIN: float = 0.05

INSTALL_HINT: str = (
    "Auto-boxing backend '{backend}' is unavailable: {package} is not installed. "
    "Install the 'boxing' extra: pip install 'trashmonkey[boxing]'. "
    "It pins autodistill, autodistill-grounding-dino and rembg[cpu] (the cpu/gpu "
    "extra of rembg supplies the onnxruntime that runs the BiRefNet session)."
)

Method = Literal["dino", "birefnet", "centerbox"]

# Pixel-space box, (x1, y1, x2, y2).
XYXY = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    """One detector hit in pixel coordinates."""

    xyxy: XYXY
    confidence: float


@dataclass
class BoxRecord:
    """Provenance for one auto-boxed image (one JSONL line)."""

    image: str
    source: str
    method: Method
    confidence: float | None
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "source": self.source,
            "method": self.method,
            "confidence": self.confidence,
            "flags": self.flags,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


# Backend interfaces. Tests inject fakes; production builds lazy real ones.
DinoPredictFn = Callable[[Path], Sequence[Detection]]
MaskFn = Callable[[Path], npt.NDArray[np.uint8]]
# progress(done, total, image_path) after each processed image.
ProgressFn = Callable[[int, int, Path], None]
