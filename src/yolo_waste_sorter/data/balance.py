"""Cap-based class balancing AFTER dedup (T4).

Big classes are capped (default 1,500 images per target class); small ones
are NEVER duplicated or oversampled. Sampling is seeded (42), uniform without
replacement, and stratified by source so no source is wiped out of a class.
A class landing under the 800 floor logs a warning -- never an error.
Sources held out for TEST-1 (T6 leave-one-source-out) bypass capping
entirely: TEST-1 must contain ALL post-dedup images of that source.
Per-source per-class caps from configs/datasets.yaml (``cap``) clamp a
source's pool before the global cap.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from yolo_waste_sorter.data.dedup import Item

logger = logging.getLogger(__name__)

DEFAULT_CAP = 1500  # max images per target class after dedup (T4)
FLOOR = 800  # warn (never error) when a class lands below this


class BalanceError(Exception):
    """Balance inputs are malformed."""


@dataclass(frozen=True)
class BalanceResult:
    """Outcome of balance_items: kept items plus per-class-per-source counts."""

    kept: tuple[Item, ...]
    cap: int
    floor: int
    seed: int
    exempt_sources: tuple[str, ...]
    counts: dict[str, dict[str, dict[str, int]]]  # class -> source -> {kept, dropped}
    floor_warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": "balance",
            "cap": self.cap,
            "floor": self.floor,
            "seed": self.seed,
            "exempt_sources": list(self.exempt_sources),
            "counts": self.counts,
            "floor_warnings": list(self.floor_warnings),
            "kept": [item.key for item in self.kept],
        }

    def write_manifest(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def _allocate(pool_sizes: Mapping[str, int], cap: int) -> dict[str, int]:
    """Largest-remainder proportional allocation of `cap` across sources.

    Every source with a non-empty pool gets at least one slot (when cap
    allows), so capping never wipes a source out of a class.
    """
    total = sum(pool_sizes.values())
    sources = sorted(pool_sizes)
    if total <= cap:
        return dict(pool_sizes)
    quotas = {s: cap * pool_sizes[s] / total for s in sources}
    alloc = {s: min(pool_sizes[s], int(quotas[s])) for s in sources}
    if len(sources) <= cap:
        for s in sources:
            if pool_sizes[s] > 0 and alloc[s] == 0:
                alloc[s] = 1
    remaining = cap - sum(alloc.values())
    by_remainder = sorted(sources, key=lambda s: (-(quotas[s] - int(quotas[s])), s))
    for s in by_remainder:
        if remaining <= 0:
            break
        room = pool_sizes[s] - alloc[s]
        if room > 0:
            take = min(room, remaining)
            alloc[s] += take
            remaining -= take
    return alloc


def _sample(items: Sequence[Item], n: int, rng_key: str, seed: int) -> list[Item]:
    """Seeded uniform sample WITHOUT replacement -- never duplicates."""
    ordered = sorted(items, key=lambda it: it.key)
    if n >= len(ordered):
        return list(ordered)
    rng = random.Random(f"{seed}:{rng_key}")
    return sorted(rng.sample(ordered, n), key=lambda it: it.key)


def balance_items(
    items: Iterable[Item],
    *,
    cap: int = DEFAULT_CAP,
    floor: int = FLOOR,
    seed: int = 42,
    exempt_sources: frozenset[str] = frozenset(),
    source_caps: Mapping[str, Mapping[str, int]] | None = None,
) -> BalanceResult:
    """Cap each target class at `cap` images; never duplicate; warn under `floor`."""
    if cap <= 0:
        raise BalanceError(f"cap must be positive, got {cap}")
    pools: dict[str, dict[str, list[Item]]] = {}
    exempt: list[Item] = []
    for item in items:
        if item.source in exempt_sources:
            exempt.append(item)
            continue
        pools.setdefault(item.class_name, {}).setdefault(item.source, []).append(item)

    kept: list[Item] = sorted(exempt, key=lambda it: it.key)
    counts: dict[str, dict[str, dict[str, int]]] = {}
    floor_warnings: list[str] = []
    for class_name in sorted(pools):
        by_source = pools[class_name]
        clamped: dict[str, list[Item]] = {}
        for source in sorted(by_source):
            pool = by_source[source]
            limit = (source_caps or {}).get(source, {}).get(class_name, len(pool))
            clamped[source] = _sample(pool, limit, f"srccap:{class_name}:{source}", seed)
        alloc = _allocate({s: len(p) for s, p in clamped.items()}, cap)
        class_total = 0
        for source in sorted(clamped):
            chosen = _sample(clamped[source], alloc[source], f"cap:{class_name}:{source}", seed)
            kept.extend(chosen)
            class_total += len(chosen)
            counts.setdefault(class_name, {})[source] = {
                "kept": len(chosen),
                "dropped": len(by_source[source]) - len(chosen),
            }
        if class_total < floor:
            message = (
                f"class '{class_name}' has only {class_total} images after balancing "
                f"(floor {floor}) -- consider reserve sources (T4)"
            )
            floor_warnings.append(message)
            logger.warning("balance: %s", message)

    kept.sort(key=lambda it: it.key)
    if len({item.key for item in kept}) != len(kept):
        raise BalanceError("balance produced duplicate items -- this is a bug")
    return BalanceResult(
        kept=tuple(kept),
        cap=cap,
        floor=floor,
        seed=seed,
        exempt_sources=tuple(sorted(exempt_sources)),
        counts=counts,
        floor_warnings=tuple(floor_warnings),
    )
