"""Automated checks over 100% of auto-generated boxes (T3 QA layer)."""

from collections.abc import Mapping
from pathlib import Path
from statistics import mean, pstdev

from .boxes import Box, parse_label_file
from .report import ImageQA, ProvenanceRecord, QAReport, load_provenance

# Check thresholds (T3 decision). Named constants, not magic numbers.
AREA_RATIO_MIN = 0.05  # box covering <5% of the image is suspicious
AREA_RATIO_MAX = 0.95  # box covering >95% is near-full-frame
ZSCORE_MAX = 3.0  # per-class area / aspect-ratio outlier bar
EDGE_TOUCH_EPS = 0.005  # normalized distance counted as touching a border
EDGE_TOUCH_MIN = 3  # touching >=3 of 4 borders flags the box
CONFIDENCE_MIN = 0.30  # autobox provenance confidence below this flags

FLAG_BOX_COUNT = "box_count"
FLAG_AREA_EXTREME = "area_extreme"
FLAG_AREA_ZSCORE = "area_zscore"
FLAG_ASPECT_ZSCORE = "aspect_zscore"
FLAG_EDGE_CONTACT = "edge_contact"
FLAG_LOW_CONFIDENCE = "low_confidence"
FLAG_CENTERBOX = "centerbox"

ALL_FLAGS = (
    FLAG_BOX_COUNT,
    FLAG_AREA_EXTREME,
    FLAG_AREA_ZSCORE,
    FLAG_ASPECT_ZSCORE,
    FLAG_EDGE_CONTACT,
    FLAG_LOW_CONFIDENCE,
    FLAG_CENTERBOX,
)


def _class_stats(values: Mapping[int, list[float]]) -> dict[int, tuple[float, float]]:
    """Per-class (mean, population std). Classes with <2 samples are omitted."""
    return {cls: (mean(vals), pstdev(vals)) for cls, vals in values.items() if len(vals) >= 2}


def _zscore_flagged(value: float, stats: tuple[float, float] | None) -> bool:
    if stats is None:
        return False  # undefined distribution: a lone box cannot be its own outlier
    mu, sigma = stats
    if sigma == 0.0:
        return False
    return abs(value - mu) / sigma > ZSCORE_MAX


def _box_flags(
    box: Box,
    area_stats: tuple[float, float] | None,
    aspect_stats: tuple[float, float] | None,
) -> list[str]:
    flags: list[str] = []
    if box.area < AREA_RATIO_MIN or box.area > AREA_RATIO_MAX:
        flags.append(FLAG_AREA_EXTREME)
    if _zscore_flagged(box.area, area_stats):
        flags.append(FLAG_AREA_ZSCORE)
    if box.h > 0.0 and _zscore_flagged(box.aspect, aspect_stats):
        flags.append(FLAG_ASPECT_ZSCORE)
    if box.edges_touched(EDGE_TOUCH_EPS) >= EDGE_TOUCH_MIN:
        flags.append(FLAG_EDGE_CONTACT)
    return flags


def run_checks(
    labels_dir: Path,
    provenance: Path | Mapping[str, ProvenanceRecord],
) -> QAReport:
    """Run every automated check on every labeled image.

    `provenance` is the autobox JSONL path (or an already-loaded mapping keyed
    by image stem); it is the authoritative list of images to check.
    """
    records = load_provenance(provenance) if isinstance(provenance, Path) else dict(provenance)
    if not records:
        raise ValueError(f"empty provenance for {labels_dir}")
    if not labels_dir.is_dir():
        raise NotADirectoryError(f"labels dir does not exist: {labels_dir}")
    stray = {p.stem for p in labels_dir.glob("*.txt")} - records.keys()
    if stray:
        raise ValueError(
            f"label files without provenance in {labels_dir}: {sorted(stray)[:5]}"
        )

    boxes_by_stem: dict[str, list[Box]] = {}
    areas_by_class: dict[int, list[float]] = {}
    aspects_by_class: dict[int, list[float]] = {}
    for stem in records:
        label_path = labels_dir / f"{stem}.txt"
        boxes = parse_label_file(label_path) if label_path.is_file() else []
        boxes_by_stem[stem] = boxes
        if len(boxes) == 1:  # population stats only over well-formed single-box images
            box = boxes[0]
            areas_by_class.setdefault(box.class_id, []).append(box.area)
            if box.h > 0.0:
                aspects_by_class.setdefault(box.class_id, []).append(box.aspect)

    area_stats = _class_stats(areas_by_class)
    aspect_stats = _class_stats(aspects_by_class)

    images: dict[str, ImageQA] = {}
    for stem, record in records.items():
        boxes = boxes_by_stem[stem]
        flags: list[str] = []
        if len(boxes) != 1:
            flags.append(FLAG_BOX_COUNT)
        else:
            box = boxes[0]
            flags.extend(
                _box_flags(box, area_stats.get(box.class_id), aspect_stats.get(box.class_id))
            )
        if record.confidence < CONFIDENCE_MIN:
            flags.append(FLAG_LOW_CONFIDENCE)
        if record.method == "centerbox":
            flags.append(FLAG_CENTERBOX)
        images[stem] = ImageQA(
            image=record.image,
            stem=stem,
            source=record.source,
            method=record.method,
            confidence=record.confidence,
            n_boxes=len(boxes),
            class_id=boxes[0].class_id if boxes else None,
            flags=flags,
        )
    return QAReport(labels_dir=str(labels_dir), images=images)
