"""Cross-source perceptual-hash dedup (T10).

Consumes the remap stage's interim layout
(``data/interim/remapped/<class>/<source>__<name>`` with an optional YOLO
label ``.txt`` alongside) and drops exact duplicates: pHash Hamming distance
<= 2 keeps the HIGHER-priority source's copy, where priority is the order
sources appear in ``configs/datasets.yaml``. The near-duplicate band
(distance 3-8) is recorded as edges for instance grouping (T6) -- a
connected component of near-dups is "the same physical object" and never
straddles a split boundary. A per-source-pair overlap matrix is emitted in
the manifest for the paper.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

EXACT_THRESHOLD = 2  # Hamming distance <= 2 -> exact duplicate
NEAR_BAND = (3, 8)  # distance in [3, 8] -> same physical object (instance group)
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


class DedupError(Exception):
    """The interim layout or dedup inputs are malformed."""


@dataclass(frozen=True)
class Item:
    """One remapped image: key is ``<class>/<source>__<name>`` (unique)."""

    key: str
    class_name: str
    source: str
    image: Path
    label: Path | None


@dataclass(frozen=True)
class DroppedDup:
    """An exact duplicate removed in favour of a higher-priority source."""

    key: str
    duplicate_of: str
    distance: int


@dataclass(frozen=True)
class NearEdge:
    """A near-duplicate pair (distance 3-8) between two KEPT images."""

    key_a: str
    key_b: str
    distance: int


def scan_remapped(root: Path, classes: Sequence[str]) -> list[Item]:
    """List remapped items under root/<class>/<source>__<name>, sorted by key."""
    if not root.is_dir():
        raise DedupError(f"remapped root not found: {root}")
    items: list[Item] = []
    for class_name in classes:
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise DedupError(f"missing class directory: {class_dir}")
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if "__" not in path.name:
                raise DedupError(
                    f"{path}: filename must be '<source>__<name>' (remap-stage convention)"
                )
            source = path.name.split("__", 1)[0]
            label = path.with_suffix(".txt")
            items.append(
                Item(
                    key=f"{class_name}/{path.name}",
                    class_name=class_name,
                    source=source,
                    image=path,
                    label=label if label.is_file() else None,
                )
            )
    return items


def phash(path: Path) -> int:
    """64-bit perceptual hash of an image as an int."""
    import imagehash
    from PIL import Image

    with Image.open(path) as img:
        return int(str(imagehash.phash(img)), 16)


@dataclass(frozen=True)
class DedupResult:
    """Outcome of dedup_items, written to the dedup manifest."""

    kept: tuple[Item, ...]
    dropped: tuple[DroppedDup, ...]
    near_edges: tuple[NearEdge, ...]
    overlap_matrix: dict[str, dict[str, int]]  # kept source -> dropped source -> count
    near_matrix: dict[str, dict[str, int]]  # source pair -> near-dup edge count
    source_priority: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": "dedup",
            "method": "phash",
            "exact_threshold": EXACT_THRESHOLD,
            "near_band": list(NEAR_BAND),
            "source_priority": list(self.source_priority),
            "kept": [item.key for item in self.kept],
            "dropped": [
                {"image": d.key, "duplicate_of": d.duplicate_of, "distance": d.distance}
                for d in self.dropped
            ],
            "near_edges": [[e.key_a, e.key_b, e.distance] for e in self.near_edges],
            "overlap_matrix": self.overlap_matrix,
            "near_matrix": self.near_matrix,
        }

    def write_manifest(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def _bump(matrix: dict[str, dict[str, int]], src_a: str, src_b: str) -> None:
    row = matrix.setdefault(src_a, {})
    row[src_b] = row.get(src_b, 0) + 1


def dedup_items(
    items: Iterable[Item],
    source_priority: Sequence[str],
    *,
    hash_fn: Callable[[Path], int] = phash,
) -> DedupResult:
    """Drop exact duplicates by source priority; record near-dup edges.

    Items are processed in (priority, key) order, so on an exact match the
    kept copy always belongs to the earlier source in ``source_priority``
    (ties within one source keep the lexicographically smaller key).
    """
    priority = {name: rank for rank, name in enumerate(source_priority)}
    ordered = sorted(items, key=lambda it: (priority.get(it.source, -1), it.key))
    for item in ordered:
        if item.source not in priority:
            raise DedupError(
                f"{item.key}: source '{item.source}' not in the registry priority list "
                f"{list(source_priority)}"
            )

    kept: list[tuple[Item, int]] = []
    dropped: list[DroppedDup] = []
    near_edges: list[NearEdge] = []
    overlap: dict[str, dict[str, int]] = {}
    near: dict[str, dict[str, int]] = {}
    lo, hi = NEAR_BAND
    for item in ordered:
        digest = hash_fn(item.image)
        match: tuple[Item, int] | None = None
        edges: list[tuple[Item, int]] = []
        for other, other_digest in kept:
            distance = (digest ^ other_digest).bit_count()
            if distance <= EXACT_THRESHOLD:
                match = (other, distance)
                break
            if lo <= distance <= hi:
                edges.append((other, distance))
        if match is not None:
            other, distance = match
            dropped.append(
                DroppedDup(key=item.key, duplicate_of=other.key, distance=distance)
            )
            _bump(overlap, other.source, item.source)
            continue
        for other, distance in edges:
            near_edges.append(NearEdge(key_a=other.key, key_b=item.key, distance=distance))
            _bump(near, other.source, item.source)
        kept.append((item, digest))

    logger.info(
        "dedup: kept %d, dropped %d exact duplicates, %d near-dup edges",
        len(kept),
        len(dropped),
        len(near_edges),
    )
    return DedupResult(
        kept=tuple(item for item, _ in kept),
        dropped=tuple(dropped),
        near_edges=tuple(near_edges),
        overlap_matrix=overlap,
        near_matrix=near,
        source_priority=tuple(source_priority),
    )
