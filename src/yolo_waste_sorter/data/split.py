"""Instance-grouped, stratified train/val split + leave-one-source-out TEST-1 (T6).

TEST-1 is ALL images of ``eval.leave_out_source`` (config; 'realwaste' after
the census), excluded from train/val entirely; a null value warns loudly and
skips TEST-1. The remaining pool splits train/val stratified by source x
class on top of instance groups: connected components over the dedup stage's
near-duplicate graph are "the same physical object" and NEVER straddle a
split boundary. ``emit_dataset`` writes the final YOLO detect layout
(``data/processed/<experiment>/{images,labels}/{train,val,test}/``) plus
``dataset.yaml`` with names in config class order.
"""

from __future__ import annotations

import logging
import random
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from yolo_waste_sorter.data.dedup import Item, NearEdge

logger = logging.getLogger(__name__)

DEFAULT_VAL_FRACTION = 0.15
SPLITS = ("train", "val", "test")


class SplitError(Exception):
    """Split inputs or the emitted dataset are malformed."""


@dataclass(frozen=True)
class SplitResult:
    """Split assignment per image plus instance-group ids and counts."""

    assignments: dict[str, str]  # image key -> train|val|test
    group_ids: dict[str, str]  # image key -> instance-group id (min member key)
    counts: dict[str, dict[str, int]]  # split -> class -> count
    seed: int
    val_fraction: float
    leave_out_source: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": "split",
            "seed": self.seed,
            "val_fraction": self.val_fraction,
            "leave_out_source": self.leave_out_source,
            "counts": self.counts,
            "groups": dict(sorted(self.group_ids.items())),
            "assignments": dict(sorted(self.assignments.items())),
        }

    def write_manifest(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def group_instances(keys: Iterable[str], edges: Iterable[NearEdge]) -> dict[str, str]:
    """Union-find over the near-dup graph; group id = smallest member key."""
    parent: dict[str, str] = {key: key for key in keys}

    def find(key: str) -> str:
        root = key
        while parent[root] != root:
            root = parent[root]
        while parent[key] != root:
            parent[key], key = root, parent[key]
        return root

    for edge in edges:
        if edge.key_a in parent and edge.key_b in parent:
            root_a, root_b = find(edge.key_a), find(edge.key_b)
            if root_a != root_b:
                parent[max(root_a, root_b)] = min(root_a, root_b)
    return {key: find(key) for key in parent}


def _primary_stratum(members: Sequence[Item]) -> tuple[str, str]:
    """Majority (source, class) of a group; ties break lexicographically."""
    tally: dict[tuple[str, str], int] = {}
    for item in members:
        stratum = (item.source, item.class_name)
        tally[stratum] = tally.get(stratum, 0) + 1
    return min(tally, key=lambda s: (-tally[s], s))


def split_items(
    items: Iterable[Item],
    edges: Iterable[NearEdge],
    *,
    leave_out_source: str | None,
    val_fraction: float | None,
    seed: int = 42,
) -> SplitResult:
    """Assign every item to train/val/test; instance groups never straddle."""
    if val_fraction is None:
        logger.info("split: eval.val_fraction is null, using default %.2f", DEFAULT_VAL_FRACTION)
        val_fraction = DEFAULT_VAL_FRACTION
    if not 0.0 < val_fraction < 1.0:
        raise SplitError(f"val_fraction must be in (0, 1), got {val_fraction}")

    by_key = {item.key: item for item in sorted(items, key=lambda it: it.key)}
    assignments: dict[str, str] = {}
    if leave_out_source is None:
        logger.warning(
            "split: eval.leave_out_source is null -- TEST-1 (leave-one-source-out) is "
            "SKIPPED; the dataset will have no test split. Set it before any real run."
        )
        pool = list(by_key.values())
    else:
        test = [it for it in by_key.values() if it.source == leave_out_source]
        if not test:
            raise SplitError(
                f"leave_out_source '{leave_out_source}' has no images in the input pool"
            )
        for item in test:
            assignments[item.key] = "test"
        pool = [it for it in by_key.values() if it.source != leave_out_source]

    group_ids = group_instances((it.key for it in pool), edges)
    groups: dict[str, list[Item]] = {}
    for item in pool:
        groups.setdefault(group_ids[item.key], []).append(item)

    strata_sizes: dict[tuple[str, str], int] = {}
    for item in pool:
        stratum = (item.source, item.class_name)
        strata_sizes[stratum] = strata_sizes.get(stratum, 0) + 1
    targets = {s: int(n * val_fraction + 0.5) for s, n in strata_sizes.items()}
    val_counts = {s: 0 for s in strata_sizes}

    by_stratum: dict[tuple[str, str], list[str]] = {}
    for gid, members in groups.items():
        by_stratum.setdefault(_primary_stratum(members), []).append(gid)
    for stratum in sorted(by_stratum):
        gids = sorted(by_stratum[stratum])
        random.Random(f"{seed}:split:{stratum[0]}:{stratum[1]}").shuffle(gids)
        for gid in gids:
            to_val = val_counts[stratum] < targets[stratum]
            for member in groups[gid]:
                assignments[member.key] = "val" if to_val else "train"
                if to_val:
                    member_stratum = (member.source, member.class_name)
                    val_counts[member_stratum] += 1

    counts: dict[str, dict[str, int]] = {}
    for key, split in assignments.items():
        class_name = by_key[key].class_name
        per_class = counts.setdefault(split, {})
        per_class[class_name] = per_class.get(class_name, 0) + 1
    counts = {s: dict(sorted(c.items())) for s, c in sorted(counts.items())}
    return SplitResult(
        assignments=assignments,
        group_ids=group_ids,
        counts=counts,
        seed=seed,
        val_fraction=val_fraction,
        leave_out_source=leave_out_source,
    )


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
