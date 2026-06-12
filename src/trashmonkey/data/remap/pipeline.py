"""Remap orchestration: plan (fail fast) -> clean previous outputs -> copy -> manifest.

Outputs per source:
  data/interim/remapped/<class>/<source>__<orig_name>   mapped images
  (+ rewritten YOLO label txt alongside, det sources only)
  data/interim/wilderness/<source>__<orig_name>          DROP-routed images (T9 probe)
  data/interim/remapped/.manifests/<source>.yaml         RemapManifest
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from trashmonkey.data.download.registry import DROP, SourceSpec
from trashmonkey.data.remap.classification import plan_classification
from trashmonkey.data.remap.detection import plan_detection
from trashmonkey.data.remap.errors import RemapError
from trashmonkey.data.remap.layout import DestAllocator, clean_previous, copy_into
from trashmonkey.data.remap.manifest import (
    REMAPPED_DIRNAME,
    WILDERNESS_DIRNAME,
    RemapManifest,
    manifest_path,
)
from trashmonkey.data.remap.names import resolve_class_names

# (raw image, target class or None for wilderness, rewritten label lines or None)
_PlanItem = tuple[Path, str | None, tuple[str, ...] | None]


def _label_lookup(spec: SourceSpec) -> dict[str, str]:
    """Case-insensitive source label -> target class (or DROP), drops folded in."""
    lookup: dict[str, str] = {}
    for label, target in spec.mapping.items():
        key = label.lower()
        if key in lookup and lookup[key] != target:
            raise RemapError(
                f"source '{spec.name}': mapping keys collide case-insensitively on "
                f"'{key}' with different targets"
            )
        lookup[key] = target
    for label in spec.drops:
        lookup.setdefault(label.lower(), DROP)
    return lookup


def _plan(
    spec: SourceSpec,
    source_root: Path,
    lookup: dict[str, str],
    target_index: dict[str, int],
    names: Sequence[str] | None,
) -> tuple[list[_PlanItem], list[str], list[str], int]:
    """Returns (plan items, skipped, errors, dropped box count)."""
    if spec.annotation_type == "cls":
        cls_items, skipped = plan_classification(source_root, spec.name, lookup)
        plan: list[_PlanItem] = [
            (item.src, None if item.target == DROP else item.target, None) for item in cls_items
        ]
        return plan, skipped, [], 0
    det_names = tuple(names) if names is not None else resolve_class_names(source_root, spec.name)
    det_items, skipped, errors = plan_detection(
        source_root, spec.name, lookup, det_names, target_index
    )
    plan = [
        (item.src, item.target, item.lines if item.target is not None else None)
        for item in det_items
    ]
    return plan, skipped, errors, sum(item.dropped_boxes for item in det_items)


def remap_source(
    spec: SourceSpec,
    raw_root: Path,
    interim_root: Path,
    target_classes: Sequence[str],
    *,
    names: Sequence[str] | None = None,
) -> RemapManifest:
    """Remap one source from data/raw/<name>/ into the unified interim layout.

    Copies, never moves: data/raw/ stays byte-identical. Rewritten det class
    ids follow target_classes order (configs/config.yaml `classes`). `names`
    overrides class-index discovery for det sources (use it once the registry
    carries an explicit list). Idempotent per source: previous outputs under
    the `<source>__` prefix are removed before copying.
    """
    source_root = raw_root / spec.name
    if not source_root.is_dir():
        raise RemapError(f"source '{spec.name}': raw tree not found at {source_root}")
    lookup = _label_lookup(spec)
    bad = sorted({t for t in lookup.values() if t != DROP and t not in set(target_classes)})
    if bad:
        raise RemapError(
            f"source '{spec.name}': mapping targets {bad} are not in the "
            f"target classes {list(target_classes)}"
        )
    target_index = {cls: i for i, cls in enumerate(target_classes)}

    plan, skipped, errors, dropped_boxes = _plan(spec, source_root, lookup, target_index, names)

    clean_previous(interim_root, spec.name)
    allocator = DestAllocator(spec.name)
    class_counts = {cls: 0 for cls in target_classes}
    drop_count = 0
    for src, target, lines in plan:
        if target is None:
            dest_dir = interim_root / WILDERNESS_DIRNAME
            drop_count += 1
        else:
            dest_dir = interim_root / REMAPPED_DIRNAME / target
            class_counts[target] += 1
        dest = allocator.allocate(dest_dir, src.name, with_label=lines is not None)
        copy_into(src, dest)
        if lines is not None:
            dest.with_suffix(".txt").write_text("\n".join(lines) + "\n")

    manifest = RemapManifest(
        source=spec.name,
        annotation_type=spec.annotation_type,
        class_counts=class_counts,
        drop_count=drop_count,
        dropped_boxes=dropped_boxes,
        skipped=tuple(skipped),
        errors=tuple(errors),
    )
    manifest.write(manifest_path(interim_root, spec.name))
    return manifest


def remap_sources(
    specs: Iterable[SourceSpec],
    raw_root: Path,
    interim_root: Path,
    target_classes: Sequence[str],
    *,
    names: Mapping[str, Sequence[str]] | None = None,
) -> list[RemapManifest]:
    """Remap every source in order; fail fast on the first error."""
    by_source = dict(names) if names is not None else {}
    return [
        remap_source(spec, raw_root, interim_root, target_classes, names=by_source.get(spec.name))
        for spec in specs
    ]
