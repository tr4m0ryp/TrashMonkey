"""Pure label-quality filter (T4): drop low-quality auto-labels.

Given the remapped :class:`~trashmonkey.data.dedup.Item` list plus the
already-loaded autobox provenance (stem -> :class:`ProvenanceRecord`) and a set
of thresholds, decide which items to keep. An item is dropped when its
provenance says it came from a fallback ``centerbox`` method, when its
confidence is below the bar, or when its YOLO box geometry is degenerate
(whole-frame artifact or speck). Items with no provenance entry are detector
sources that were never auto-boxed: they are KEPT unless their own shipped
box is clearly degenerate.

This module is side-effect free apart from *reading* the label files referenced
by ``Item.label``. It loads no config and no provenance from disk -- the caller
(T5 pipeline wiring) supplies both. Parsing is deliberately defensive: any
ambiguity (missing file, empty file, malformed line, read error) KEEPS the item,
because a false drop silently loses training data.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Set
from dataclasses import dataclass
from pathlib import Path

from ..dedup import Item
from ..qa.report import ProvenanceRecord

# Drop-reason codes (named constants, not magic strings, for the manifest).
REASON_METHOD = "drop_method"  # provenance method in drop_methods (e.g. centerbox)
REASON_LOW_CONFIDENCE = "low_confidence"  # provenance confidence below min_confidence
REASON_BOX_TOO_LARGE = "box_too_large"  # largest box area fraction above max_box_frac
REASON_BOX_TOO_SMALL = "box_too_small"  # largest box area fraction below min_box_frac

ALL_REASONS = (
    REASON_METHOD,
    REASON_LOW_CONFIDENCE,
    REASON_BOX_TOO_LARGE,
    REASON_BOX_TOO_SMALL,
)


@dataclass(frozen=True)
class FilterResult:
    """Outcome of :func:`filter_items`, suitable for a drop manifest.

    ``reasons`` maps each dropped item's key to the single reason code that
    caused the drop (precedence: method -> confidence -> geometry), so a
    manifest can attribute every removal.
    """

    kept: tuple[Item, ...]
    dropped: tuple[Item, ...]
    reasons: dict[str, str]


def _largest_box_frac(label: Path | None) -> float | None:
    """Largest YOLO box area fraction (w*h) in ``label``, or None if ambiguous.

    Returns None -- meaning "do not geometry-judge this item" -- whenever the
    label is absent, unreadable, empty, or contains any malformed/out-of-range
    line. Conservative by design: geometry only drops items whose largest box is
    unambiguously parseable and degenerate.
    """
    if label is None:
        return None
    try:
        text = label.read_text()
    except OSError:
        return None
    largest: float | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            return None  # malformed line -> ambiguous, keep the item
        try:
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            return None
        if not (0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
            return None  # out-of-range geometry -> ambiguous, keep
        frac = w * h
        if largest is None or frac > largest:
            largest = frac
    return largest


def _drop_reason(
    item: Item,
    record: ProvenanceRecord | None,
    *,
    drop_methods: Set[str],
    min_confidence: float,
    max_box_frac: float,
    min_box_frac: float,
) -> str | None:
    """Single drop reason for one item, or None to keep it.

    Precedence: method, then confidence, then geometry. Method and confidence
    apply only when the item has a provenance record; geometry applies to every
    item with a parseable largest box (det sources included, but only when the
    box is unambiguously degenerate).
    """
    if record is not None:
        if record.method in drop_methods:
            return REASON_METHOD
        if record.confidence is not None and record.confidence < min_confidence:
            return REASON_LOW_CONFIDENCE
    frac = _largest_box_frac(item.label)
    if frac is not None:
        if frac > max_box_frac:
            return REASON_BOX_TOO_LARGE
        if frac < min_box_frac:
            return REASON_BOX_TOO_SMALL
    return None


def filter_items(
    items: Iterable[Item],
    provenance: Mapping[str, ProvenanceRecord],
    *,
    drop_methods: Set[str],
    min_confidence: float,
    max_box_frac: float,
    min_box_frac: float,
) -> FilterResult:
    """Drop low-quality auto-labels; keep everything else.

    Parameters
    ----------
    items:
        Remapped items to filter. Iterated once; output order follows input.
    provenance:
        Already-loaded autobox provenance keyed by image filename stem
        (``Item.image`` / label stem). Items with no entry are detector sources
        that were never auto-boxed and are kept unless clearly degenerate.
    drop_methods:
        Provenance methods that mark a fallback label to drop (e.g.
        ``{"centerbox"}``).
    min_confidence:
        Drop an item whose provenance confidence is not ``None`` and strictly
        below this bar.
    max_box_frac, min_box_frac:
        Keep an item only if its largest YOLO box area fraction is within
        ``[min_box_frac, max_box_frac]``; above is a whole-frame artifact, below
        is a speck. Items whose geometry cannot be parsed unambiguously are kept.

    Returns
    -------
    FilterResult
        Kept items, dropped items, and a key -> reason-code map. Deterministic:
        no RNG; identical inputs yield identical outputs.
    """
    kept: list[Item] = []
    dropped: list[Item] = []
    reasons: dict[str, str] = {}
    for item in items:
        record = provenance.get(item.label.stem) if item.label is not None else None
        reason = _drop_reason(
            item,
            record,
            drop_methods=drop_methods,
            min_confidence=min_confidence,
            max_box_frac=max_box_frac,
            min_box_frac=min_box_frac,
        )
        if reason is None:
            kept.append(item)
        else:
            dropped.append(item)
            reasons[item.key] = reason
    return FilterResult(kept=tuple(kept), dropped=tuple(dropped), reasons=reasons)
