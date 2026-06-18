"""Assign every item to a split; instance groups never straddle a boundary.

Order of carving (each tier removed from the pool before the next):

1. TEST-1 -- ALL images of ``leave_out_source`` (T6 leave-one-source-out); a
   null value warns loudly and skips the tier.
2. ``wild_test`` -- ALL images of any ``test_only`` source (``role: test_only``
   in the registry, e.g. the wild garbage-detection set), excluded from
   train/val AND from balancing upstream.
3. ``clean_test`` -- a group-aware, (source, class)-stratified ``clean_holdout``
   fraction of instance groups whose source is in ``clean_holdout.sources``.
4. The remainder splits train/val stratified by source x class on top of
   instance groups.

When the new knobs are inert (no test_only sources, fraction 0/None) the result
is byte-identical to the legacy three-tier split.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterable, Sequence, Set

from trashmonkey.data.dedup import Item, NearEdge

from .grouping import group_instances, primary_stratum
from .result import DEFAULT_VAL_FRACTION, SplitError, SplitResult

logger = logging.getLogger(__name__)


def _carve_clean(
    groups: dict[str, list[Item]],
    *,
    sources: Set[str],
    fraction: float,
    seed: int,
) -> set[str]:
    """Group ids carved into clean_test: a stratified `fraction` per (source,class).

    Only groups whose primary stratum's source is in ``sources`` are eligible.
    Eligible groups are bucketed by primary stratum, shuffled with the existing
    seeded ``random.Random`` pattern, and the first ``round(n*fraction)`` of each
    bucket are taken -- so the carve is deterministic and group-aware.
    """
    eligible: dict[tuple[str, str], list[str]] = {}
    for gid, members in groups.items():
        stratum = primary_stratum(members)
        if stratum[0] in sources:
            eligible.setdefault(stratum, []).append(gid)
    carved: set[str] = set()
    for stratum in sorted(eligible):
        gids = sorted(eligible[stratum])
        random.Random(f"{seed}:clean_holdout:{stratum[0]}:{stratum[1]}").shuffle(gids)
        take = int(len(gids) * fraction + 0.5)
        carved.update(gids[:take])
    return carved


def _trainval(
    pool: Sequence[Item],
    group_ids: dict[str, str],
    groups: dict[str, list[Item]],
    skip_groups: Set[str],
    *,
    val_fraction: float,
    seed: int,
) -> dict[str, str]:
    """Stratified train/val over the groups not carved into another tier.

    Strata sizes / val targets are computed over the train/val pool only (groups
    in ``skip_groups`` are excluded), so a carve shrinks the pool exactly as a
    leave-out would. With no skips this is the legacy allocation verbatim.
    """
    tv_items = [it for it in pool if group_ids[it.key] not in skip_groups]
    strata_sizes: dict[tuple[str, str], int] = {}
    for item in tv_items:
        stratum = (item.source, item.class_name)
        strata_sizes[stratum] = strata_sizes.get(stratum, 0) + 1
    targets = {s: int(n * val_fraction + 0.5) for s, n in strata_sizes.items()}
    val_counts = {s: 0 for s in strata_sizes}

    by_stratum: dict[tuple[str, str], list[str]] = {}
    for gid, members in groups.items():
        if gid in skip_groups:
            continue
        by_stratum.setdefault(primary_stratum(members), []).append(gid)

    assignments: dict[str, str] = {}
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
    return assignments


def split_items(
    items: Iterable[Item],
    edges: Iterable[NearEdge],
    *,
    leave_out_source: str | None,
    val_fraction: float | None,
    seed: int = 42,
    test_only_sources: Set[str] = frozenset(),
    clean_holdout_sources: Set[str] = frozenset(),
    clean_holdout_fraction: float | None = None,
) -> SplitResult:
    """Assign every item to a split; instance groups never straddle."""
    if val_fraction is None:
        logger.info("split: eval.val_fraction is null, using default %.2f", DEFAULT_VAL_FRACTION)
        val_fraction = DEFAULT_VAL_FRACTION
    if not 0.0 < val_fraction < 1.0:
        raise SplitError(f"val_fraction must be in (0, 1), got {val_fraction}")

    by_key = {item.key: item for item in sorted(items, key=lambda it: it.key)}
    assignments: dict[str, str] = {}

    remaining = list(by_key.values())
    if leave_out_source is not None:
        test = [it for it in remaining if it.source == leave_out_source]
        if not test:
            raise SplitError(
                f"leave_out_source '{leave_out_source}' has no images in the input pool"
            )
        for item in test:
            assignments[item.key] = "test"
        remaining = [it for it in remaining if it.source != leave_out_source]
    else:
        logger.warning(
            "split: eval.leave_out_source is null -- TEST-1 (leave-one-source-out) is "
            "SKIPPED; the dataset will have no test split. Set it before any real run."
        )

    if test_only_sources:
        for item in remaining:
            if item.source in test_only_sources:
                assignments[item.key] = "wild_test"
        remaining = [it for it in remaining if it.source not in test_only_sources]

    pool = remaining
    group_ids = group_instances((it.key for it in pool), edges)
    groups: dict[str, list[Item]] = {}
    for item in pool:
        groups.setdefault(group_ids[item.key], []).append(item)

    carved: set[str] = set()
    if clean_holdout_sources and clean_holdout_fraction:
        carved = _carve_clean(
            groups,
            sources=clean_holdout_sources,
            fraction=clean_holdout_fraction,
            seed=seed,
        )
        for gid in carved:
            for member in groups[gid]:
                assignments[member.key] = "clean_test"

    assignments.update(
        _trainval(pool, group_ids, groups, carved, val_fraction=val_fraction, seed=seed)
    )

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
        clean_holdout_fraction=clean_holdout_fraction,
        clean_holdout_sources=tuple(sorted(clean_holdout_sources)),
        test_only_sources=tuple(sorted(test_only_sources)),
    )
