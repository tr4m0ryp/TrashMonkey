"""Pure geometry: mask -> rect, center box, YOLO txt normalization.

No heavy dependencies; everything here is exercised directly by tests.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from trashmonkey.data.autobox.types import XYXY

_SENTINEL = np.iinfo(np.int64).max


def clamp_box(box: XYXY, width: int, height: int) -> XYXY:
    """Clamp a pixel box to image bounds; reject degenerate results."""
    x1 = min(max(box[0], 0.0), float(width))
    y1 = min(max(box[1], 0.0), float(height))
    x2 = min(max(box[2], 0.0), float(width))
    y2 = min(max(box[3], 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"degenerate box after clamping: {box} in {width}x{height}")
    return (x1, y1, x2, y2)


def yolo_line(class_id: int, box: XYXY, width: int, height: int) -> str:
    """Format one YOLO label line: 'class_id cx cy w h', normalized to [0,1]."""
    x1, y1, x2, y2 = clamp_box(box, width, height)
    cx = (x1 + x2) / 2.0 / width
    cy = (y1 + y2) / 2.0 / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def center_box(width: int, height: int, margin: float) -> XYXY:
    """Image minus `margin` (fraction of each dimension) on all sides."""
    if not 0.0 <= margin < 0.5:
        raise ValueError(f"center-box margin must be in [0, 0.5): {margin}")
    return (margin * width, margin * height, (1.0 - margin) * width, (1.0 - margin) * height)


def _propagate_min_labels(
    labels: npt.NDArray[np.int64], foreground: npt.NDArray[np.bool_]
) -> npt.NDArray[np.int64]:
    """One step of 4-neighbor minimum-label propagation over the foreground."""
    padded = np.pad(labels, 1, constant_values=_SENTINEL)
    stacked = np.stack(
        (
            labels,
            padded[:-2, 1:-1],  # up
            padded[2:, 1:-1],  # down
            padded[1:-1, :-2],  # left
            padded[1:-1, 2:],  # right
        )
    )
    out: npt.NDArray[np.int64] = stacked.min(axis=0)
    out[~foreground] = _SENTINEL
    return out


def largest_component_box(mask: npt.NDArray[np.uint8]) -> tuple[XYXY, int] | None:
    """Enclosing rect and pixel area of the largest 4-connected component.

    Accepts a uint8 mask (foreground >= 128) or a bool mask. Returns None for
    an empty mask. Labeling is vectorized min-label propagation to a fixpoint:
    exact, dependency-free, and fast enough for a fallback path.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    foreground = mask.astype(bool) if mask.dtype == np.bool_ else mask >= 128
    if not foreground.any():
        return None
    h, w = foreground.shape
    labels = np.where(foreground, np.arange(1, h * w + 1, dtype=np.int64).reshape(h, w), _SENTINEL)
    while True:
        nxt = _propagate_min_labels(labels, foreground)
        if np.array_equal(nxt, labels):
            break
        labels = nxt
    component_ids, counts = np.unique(labels[foreground], return_counts=True)
    best_id = component_ids[int(np.argmax(counts))]
    area = int(counts.max())
    rows, cols = np.nonzero(labels == best_id)
    box: XYXY = (
        float(cols.min()),
        float(rows.min()),
        float(cols.max()) + 1.0,
        float(rows.max()) + 1.0,
    )
    return box, area


def mask_to_box(
    mask: npt.NDArray[np.uint8],
    *,
    min_area_frac: float,
    max_area_frac: float,
) -> XYXY | None:
    """Largest-component enclosing rect, or None if the mask is implausible.

    The mask is rejected (returns None) when its largest connected component
    covers less than `min_area_frac` or more than `max_area_frac` of the image.
    """
    result = largest_component_box(mask)
    if result is None:
        return None
    box, area = result
    frac = area / float(mask.shape[0] * mask.shape[1])
    if frac < min_area_frac or frac > max_area_frac:
        return None
    return box
