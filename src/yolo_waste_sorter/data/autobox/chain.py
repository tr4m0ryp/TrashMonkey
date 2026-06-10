"""Auto-boxing chain (T3): Grounding DINO -> BiRefNet rect -> center box.

One YOLO txt label per image plus a provenance JSONL. The class ID comes from
the source mapping, never from the detector.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from yolo_waste_sorter.data.autobox.birefnet import build_birefnet_backend
from yolo_waste_sorter.data.autobox.dino import build_dino_backend
from yolo_waste_sorter.data.autobox.geometry import center_box, mask_to_box, yolo_line
from yolo_waste_sorter.data.autobox.types import (
    CENTER_BOX_MARGIN,
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


def _resolve_box(
    image_path: Path,
    width: int,
    height: int,
    *,
    dino_predict: DinoPredictFn,
    birefnet_mask: MaskFn,
    min_confidence: float,
    mask_min_frac: float,
    mask_max_frac: float,
    center_margin: float,
) -> tuple[XYXY, Method, float | None, list[str]]:
    """Run the three-stage chain for one image."""
    detections = list(dino_predict(image_path))
    accepted = [d for d in detections if d.confidence >= min_confidence]
    if accepted:
        best = max(accepted, key=lambda d: d.confidence)
        flags = ["multibox"] if len(detections) > 1 else []
        return best.xyxy, "dino", best.confidence, flags

    mask = birefnet_mask(image_path)
    if mask.shape != (height, width):
        raise ValueError(
            f"mask shape {mask.shape} does not match image {width}x{height}: {image_path}"
        )
    rect = mask_to_box(mask, min_area_frac=mask_min_frac, max_area_frac=mask_max_frac)
    if rect is not None:
        return rect, "birefnet", None, []

    return center_box(width, height, center_margin), "centerbox", None, ["centerbox"]


def box_directory(
    images_dir: Path,
    class_id: int,
    out_labels_dir: Path,
    *,
    class_name: str,
    source: str,
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
    `dino_predict` / `birefnet_mask` default to the real lazy backends (the
    'boxing' extra); tests inject fakes. Backends are built only when first
    needed, so the fallback model never loads if Grounding DINO covers all
    images.
    """
    if class_name not in PROMPTS:
        raise ValueError(f"unknown class {class_name!r}; expected one of {sorted(PROMPTS)}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images directory not found: {images_dir}")

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
                dino_predict=get_dino(),
                # Defer building the rembg backend until the fallback fires.
                birefnet_mask=lambda p: get_mask()(p),
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
