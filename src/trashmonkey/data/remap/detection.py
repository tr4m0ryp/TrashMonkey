"""Detection-source planning: locate YOLO labels, rewrite class ids per the mapping.

Box lines whose source label DROPs are removed; an image whose boxes all DROP
(or whose label file is empty) routes to the wilderness pool. Images keeping
boxes of several target classes are filed under the majority class (ties break
on first occurrence in the label file).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trashmonkey.data.download.registry import DROP
from trashmonkey.data.remap.errors import RemapError, UnmappedLabelError
from trashmonkey.data.remap.layout import is_image

_NAMES_FILES = frozenset({"data.yaml", "data.yml", "classes.txt"})


@dataclass(frozen=True)
class DetItem:
    """One detection image with rewritten label lines (target None -> wilderness)."""

    src: Path
    target: str | None
    lines: tuple[str, ...]
    dropped_boxes: int


def find_label_file(image: Path, source_root: Path) -> Path | None:
    """Sibling .txt first, else the YOLO images/ -> labels/ directory swap."""
    sibling = image.with_suffix(".txt")
    if sibling.is_file():
        return sibling
    parts = list(image.relative_to(source_root).parts)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i].lower() == "images":
            swapped = source_root.joinpath(*parts[:i], "labels", *parts[i + 1 :])
            candidate = swapped.with_suffix(".txt")
            return candidate if candidate.is_file() else None
    return None


def rewrite_label_lines(
    text: str,
    names: tuple[str, ...],
    lookup: dict[str, str],
    target_index: dict[str, int],
    *,
    source: str,
    label_rel: str,
) -> tuple[list[tuple[str, str]], int]:
    """Returns (kept (target_class, rewritten line) pairs, dropped box count)."""
    kept: list[tuple[str, str]] = []
    dropped = 0
    for lineno, line in enumerate(text.splitlines(), start=1):
        tokens = line.split()
        if not tokens:
            continue
        where = f"source '{source}': {label_rel}:{lineno}"
        try:
            class_id = int(tokens[0])
        except ValueError as exc:
            raise RemapError(f"{where}: non-integer class id {tokens[0]!r}") from exc
        if not 0 <= class_id < len(names):
            raise RemapError(f"{where}: class id {class_id} out of range for names {list(names)}")
        label = names[class_id]
        target = lookup.get(label.lower())
        if target is None:
            raise UnmappedLabelError(
                f"{where}: label '{label}' is neither in the mapping nor in drops "
                f"(the mapping must be total)"
            )
        if target == DROP:
            dropped += 1
            continue
        kept.append((target, " ".join([str(target_index[target]), *tokens[1:]])))
    return kept, dropped


def _route_target(kept: list[tuple[str, str]]) -> str | None:
    if not kept:
        return None
    counts: dict[str, int] = {}
    for target, _ in kept:
        counts[target] = counts.get(target, 0) + 1
    best = max(counts.values())
    for target, _ in kept:  # first occurrence breaks ties
        if counts[target] == best:
            return target
    return None  # unreachable


def plan_detection(
    source_root: Path,
    source: str,
    lookup: dict[str, str],
    names: tuple[str, ...],
    target_index: dict[str, int],
) -> tuple[list[DetItem], list[str], list[str]]:
    """Plan every image under source_root; returns (items, skipped, errors)."""
    items: list[DetItem] = []
    skipped: list[str] = []
    errors: list[str] = []
    for path in sorted(source_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(source_root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if not is_image(path):
            if path.suffix.lower() != ".txt" and path.name not in _NAMES_FILES:
                skipped.append(rel.as_posix())
            continue
        label_path = find_label_file(path, source_root)
        if label_path is None:
            errors.append(f"{rel.as_posix()}: no YOLO label file found")
            continue
        kept, dropped = rewrite_label_lines(
            label_path.read_text(),
            names,
            lookup,
            target_index,
            source=source,
            label_rel=label_path.relative_to(source_root).as_posix(),
        )
        items.append(
            DetItem(
                src=path,
                target=_route_target(kept),
                lines=tuple(line for _, line in kept),
                dropped_boxes=dropped,
            )
        )
    return items, skipped, errors
