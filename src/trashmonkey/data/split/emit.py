"""Emit the final YOLO detect dataset + dataset.yaml from a SplitResult.

Writes ``data/processed/<experiment>/{images,labels}/<split>/`` for every split
that actually has members (the ``used_splits`` rule), then a ``dataset.yaml``
whose split keys appear in ``SPLITS`` order and whose ``names`` are the config
class order. ``clean_test``/``wild_test`` are emitted exactly like the original
three tiers, but only when non-empty -- an inert run yields the legacy layout.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import yaml

from trashmonkey.data.dedup import Item

from .result import SPLITS, SplitError, SplitResult


def emit_dataset(
    result: SplitResult,
    items: Iterable[Item],
    processed_root: Path,
    experiment: str,
    classes: Sequence[str],
) -> Path:
    """Write the YOLO detect dataset + dataset.yaml; returns the yaml path."""
    by_key = {item.key: item for item in items}
    missing_labels = sorted(
        key for key in result.assignments if by_key[key].label is None
    )
    if missing_labels:
        raise SplitError(
            f"{len(missing_labels)} image(s) have no YOLO label (autobox/remap must run "
            f"first), e.g. {missing_labels[:5]}"
        )
    root = processed_root / experiment
    used_splits = sorted(set(result.assignments.values()), key=SPLITS.index)
    for split in used_splits:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    for key, split in sorted(result.assignments.items()):
        item = by_key[key]
        flat = key.replace("/", "__")
        shutil.copy2(item.image, root / "images" / split / flat)
        assert item.label is not None  # checked above
        shutil.copy2(item.label, root / "labels" / split / (Path(flat).stem + ".txt"))

    spec: dict[str, Any] = {"path": str(root.resolve())}
    for split in used_splits:
        spec[split] = f"images/{split}"
    spec["names"] = dict(enumerate(classes))
    yaml_path = root / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(spec, f, sort_keys=False)
    return yaml_path
