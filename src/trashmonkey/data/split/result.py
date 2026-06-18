"""Split result type, error, and the ordered split vocabulary.

``SPLITS`` is the canonical emission order. ``train``/``val``/``test`` are the
original three-tier layout; ``clean_test`` (a clean-presentation holdout carved
from training sources) and ``wild_test`` (``role: test_only`` sources excluded
from train/val) are the T5 additions. Only splits with members are ever emitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_VAL_FRACTION = 0.15
SPLITS = ("train", "val", "test", "clean_test", "wild_test")


class SplitError(Exception):
    """Split inputs or the emitted dataset are malformed."""


@dataclass(frozen=True)
class SplitResult:
    """Split assignment per image plus instance-group ids and counts.

    ``clean_holdout_fraction`` / ``clean_holdout_sources`` / ``test_only_sources``
    record the T5 knobs that produced this assignment; they default to inert
    values so the legacy three-tier construction (as in tests/plots) is
    unchanged.
    """

    assignments: dict[str, str]  # image key -> train|val|test|clean_test|wild_test
    group_ids: dict[str, str]  # image key -> instance-group id (min member key)
    counts: dict[str, dict[str, int]]  # split -> class -> count
    seed: int
    val_fraction: float
    leave_out_source: str | None
    clean_holdout_fraction: float | None = None
    clean_holdout_sources: tuple[str, ...] = ()
    test_only_sources: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": "split",
            "seed": self.seed,
            "val_fraction": self.val_fraction,
            "leave_out_source": self.leave_out_source,
            "clean_holdout_fraction": self.clean_holdout_fraction,
            "clean_holdout_sources": list(self.clean_holdout_sources),
            "test_only_sources": list(self.test_only_sources),
            "counts": self.counts,
            "groups": dict(sorted(self.group_ids.items())),
            "assignments": dict(sorted(self.assignments.items())),
        }

    def write_manifest(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
