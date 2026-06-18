"""Auto-boxing chain (T3): an ordered method chain ending in a center box.

The attempt order is configurable per source (`box_order`); the default is
Grounding DINO -> BiRefNet rect -> center box. Each method is tried in turn
until one yields a box, and the center box is always the terminal fallback.

One YOLO txt label per image plus a provenance JSONL. The class ID comes from
the source mapping, never from the detector.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from PIL import Image

from trashmonkey.data.autobox.birefnet import build_birefnet_backend
from trashmonkey.data.autobox.dino import build_dino_backend
from trashmonkey.data.autobox.geometry import center_box, mask_to_box, yolo_line
from trashmonkey.data.autobox.types import (
    CENTER_BOX_MARGIN,
    DEFAULT_BOX_ORDER,
    MASK_MAX_AREA_FRAC,
    MASK_MIN_AREA_FRAC,
    MIN_BOX_CONFIDENCE,
    PROMPTS,
    XYXY,
    BoxRecord,
    DinoPredictFn,
    MaskFn,
    Method,
    ProgressFn,
)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
PROVENANCE_FILENAME = "provenance.jsonl"

# A box attempt returns (box, confidence, flags) or None to fall through.
_Attempt = tuple[XYXY, float | None, list[str]] | None


def _try_dino(
    image_path: Path,
    *,
    get_dino: Callable[[], DinoPredictFn],
    min_confidence: float,
) -> _Attempt:
    detections = list(get_dino()(image_path))
    accepted = [d for d in detections if d.confidence >= min_confidence]
    if not accepted:
        return None
    best = max(accepted, key=lambda d: d.confidence)
    flags = ["multibox"] if len(detections) > 1 else []
    return best.xyxy, best.confidence, flags


def _try_birefnet(
    image_path: Path,
    width: int,
    height: int,
    *,
    get_mask: Callable[[], MaskFn],
    mask_min_frac: float,
    mask_max_frac: float,
) -> _Attempt:
    mask = get_mask()(image_path)
    if mask.shape != (height, width):
        raise ValueError(
            f"mask shape {mask.shape} does not match image {width}x{height}: {image_path}"
        )
    rect = mask_to_box(mask, min_area_frac=mask_min_frac, max_area_frac=mask_max_frac)
    if rect is None:
        return None
    return rect, None, []


def _resolve_box(
    image_path: Path,
    width: int,
    height: int,
    *,
    methods: Sequence[Method],
    get_dino: Callable[[], DinoPredictFn],
    get_mask: Callable[[], MaskFn],
    min_confidence: float,
    mask_min_frac: float,
    mask_max_frac: float,
    center_margin: float,
) -> tuple[XYXY, Method, float | None, list[str]]:
    """Try each method in `methods` order; fall back to the center box.

    A backend builder (`get_dino` / `get_mask`) is invoked only when its method
    is actually attempted, so e.g. a birefnet-first source that always succeeds
    never builds the DINO backend. The center box is the terminal fallback
    regardless of whether it appears in `methods`.
    """
    for method in methods:
        if method == "centerbox":
            continue  # handled below as the terminal fallback
        if method == "dino":
            attempt = _try_dino(
                image_path, get_dino=get_dino, min_confidence=min_confidence
            )
        else:  # "birefnet"
            attempt = _try_birefnet(
                image_path,
                width,
                height,
                get_mask=get_mask,
                mask_min_frac=mask_min_frac,
                mask_max_frac=mask_max_frac,
            )
        if attempt is not None:
            box, confidence, flags = attempt
            return box, method, confidence, flags

    return center_box(width, height, center_margin), "centerbox", None, ["centerbox"]


def box_directory(
    images_dir: Path,
    class_id: int,
    out_labels_dir: Path,
    *,
    class_name: str,
    source: str,
    box_order: Sequence[Method] | None = None,
    min_confidence: float = MIN_BOX_CONFIDENCE,
    mask_min_frac: float = MASK_MIN_AREA_FRAC,
    mask_max_frac: float = MASK_MAX_AREA_FRAC,
    center_margin: float = CENTER_BOX_MARGIN,
    dino_predict: DinoPredictFn | None = None,
    birefnet_mask: MaskFn | None = None,
    progress: ProgressFn | None = None,
) -> list[BoxRecord]:
    """Auto-box every image in `images_dir` with one `class_id` box each.

    Writes one YOLO txt per image into `out_labels_dir` and a provenance JSONL
    (`provenance.jsonl`) alongside them; returns the provenance records.
    `box_order` sets the per-source method attempt order (a subset of
    {dino, birefnet, centerbox}); falsy/None falls back to `DEFAULT_BOX_ORDER`
    (dino -> birefnet -> centerbox), so any source without an explicit order
    behaves exactly as before. `dino_predict` / `birefnet_mask` default to the
    real lazy backends (the 'boxing' extra); tests inject fakes. A backend is
    built only when its method is actually attempted, so a method never reached
    (e.g. DINO under a birefnet-first source that always succeeds) never loads.
    """
    if class_name not in PROMPTS:
        raise ValueError(f"unknown class {class_name!r}; expected one of {sorted(PROMPTS)}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images directory not found: {images_dir}")
    methods: tuple[Method, ...] = tuple(box_order) if box_order else DEFAULT_BOX_ORDER

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    def get_dino() -> DinoPredictFn:
        nonlocal dino_predict
        if dino_predict is None:
            dino_predict = build_dino_backend(PROMPTS[class_name], box_threshold=min_confidence)
        return dino_predict

    def get_mask() -> MaskFn:
        nonlocal birefnet_mask
        if birefnet_mask is None:
            birefnet_mask = build_birefnet_backend()
        return birefnet_mask

    records: list[BoxRecord] = []
    total = len(images)
    with open(out_labels_dir / PROVENANCE_FILENAME, "w", encoding="utf-8") as provenance:
        for index, image_path in enumerate(images):
            with Image.open(image_path) as img:
                width, height = img.size
            box, method, confidence, flags = _resolve_box(
                image_path,
                width,
                height,
                methods=methods,
                # Builders are passed (not called) so each backend loads only
                # when its method is actually attempted, in the configured order.
                get_dino=get_dino,
                get_mask=get_mask,
                min_confidence=min_confidence,
                mask_min_frac=mask_min_frac,
                mask_max_frac=mask_max_frac,
                center_margin=center_margin,
            )
            label_path = out_labels_dir / f"{image_path.stem}.txt"
            label_path.write_text(yolo_line(class_id, box, width, height) + "\n", encoding="utf-8")
            record = BoxRecord(
                image=image_path.name,
                source=source,
                method=method,
                confidence=confidence,
                flags=flags,
            )
            provenance.write(record.to_json() + "\n")
            records.append(record)
            if progress is not None:
                progress(index + 1, total, image_path)
    return records
