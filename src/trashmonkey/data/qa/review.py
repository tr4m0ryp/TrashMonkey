"""Review-queue emission and the stratified sampling primitive (T3 human review)."""

import csv
import random
import shutil
from collections.abc import Mapping, Sequence
from math import ceil
from pathlib import Path
from typing import TypeVar

from .report import QAReport

T = TypeVar("T")

INDEX_CSV_FIELDS = ("image", "flags", "source", "method", "confidence")


def emit_review_queue(report: QAReport, out_dir: Path) -> Path:
    """Copy (never move) every flagged image + label into out_dir/<flag>/.

    Writes out_dir/index.csv (image, flags, source, method, confidence) sorted
    by ascending confidence (most suspicious first) so a human can walk it in
    order. Returns the index path. Intended out_dir: data/interim/review/.
    """
    labels_dir = Path(report.labels_dir)
    flagged = sorted(report.flagged.values(), key=lambda r: (r.confidence, r.stem))
    out_dir.mkdir(parents=True, exist_ok=True)
    for rec in flagged:
        image_path = Path(rec.image)
        if not image_path.is_file():
            raise FileNotFoundError(f"flagged image missing on disk: {image_path}")
        label_path = labels_dir / f"{rec.stem}.txt"
        for flag in rec.flags:
            flag_dir = out_dir / flag
            flag_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, flag_dir / image_path.name)
            if label_path.is_file():  # box_count-flagged images may have no label file
                shutil.copy2(label_path, flag_dir / label_path.name)

    index_path = out_dir / "index.csv"
    with open(index_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(INDEX_CSV_FIELDS)
        for rec in flagged:
            writer.writerow(
                [rec.image, "|".join(rec.flags), rec.source, rec.method, rec.confidence]
            )
    return index_path


def _stratum_key(item: object, by: Sequence[str]) -> tuple[object, ...]:
    if isinstance(item, Mapping):
        return tuple(item[name] for name in by)
    return tuple(getattr(item, name) for name in by)


def stratified_sample(
    items: Sequence[T],
    frac: float,
    by: Sequence[str] = ("class_id", "method"),
    seed: int = 42,
) -> list[T]:
    """Sample ~frac of items per stratum, never less than one per stratum.

    Strata are tuples of the `by` attributes (or mapping keys) of each item;
    ceil-rounding guarantees rare strata stay represented, which is how the
    pipeline oversamples small groups in the 10% review draw.
    """
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    strata: dict[tuple[object, ...], list[T]] = {}
    for item in items:
        strata.setdefault(_stratum_key(item, by), []).append(item)
    rng = random.Random(seed)
    sample: list[T] = []
    for group in strata.values():
        sample.extend(rng.sample(group, ceil(frac * len(group))))
    return sample
