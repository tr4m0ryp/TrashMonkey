"""Per-source remap manifest (YAML): the counts that feed balancing and the paper.

Stored at data/interim/remapped/.manifests/<source>.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from trashmonkey.data.remap.errors import RemapError

REMAPPED_DIRNAME = "remapped"
WILDERNESS_DIRNAME = "wilderness"
MANIFESTS_DIRNAME = ".manifests"

_MANIFEST_KEYS = frozenset(
    {
        "source",
        "annotation_type",
        "class_counts",
        "drop_count",
        "dropped_boxes",
        "skipped",
        "errors",
    }
)


def manifest_path(interim_root: Path, source: str) -> Path:
    """Where the remap manifest for one source lives under data/interim/."""
    return interim_root / REMAPPED_DIRNAME / MANIFESTS_DIRNAME / f"{source}.yaml"


@dataclass(frozen=True)
class RemapManifest:
    """Outcome of remapping one source.

    class_counts: images copied per target class (every target listed, zeros kept).
    drop_count:   images routed to the wilderness pool (open-set probe, T9).
    dropped_boxes: det only -- label lines removed because their label DROPs.
    skipped:      non-image, non-label files left behind (relative paths).
    errors:       per-file problems that excluded an image (e.g. missing label txt).
    """

    source: str
    annotation_type: str
    class_counts: dict[str, int]
    drop_count: int
    dropped_boxes: int
    skipped: tuple[str, ...]
    errors: tuple[str, ...]

    def write(self, path: Path) -> None:
        payload: dict[str, Any] = {
            "source": self.source,
            "annotation_type": self.annotation_type,
            "class_counts": dict(self.class_counts),
            "drop_count": self.drop_count,
            "dropped_boxes": self.dropped_boxes,
            "skipped": list(self.skipped),
            "errors": list(self.errors),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(payload, sort_keys=False))

    @classmethod
    def read(cls, path: Path) -> RemapManifest:
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict) or set(raw) != _MANIFEST_KEYS:
            raise RemapError(f"malformed remap manifest at {path}: {raw!r}")
        counts = raw["class_counts"]
        if not isinstance(counts, dict):
            raise RemapError(f"malformed class_counts in remap manifest at {path}: {counts!r}")
        return cls(
            source=str(raw["source"]),
            annotation_type=str(raw["annotation_type"]),
            class_counts={str(k): int(v) for k, v in counts.items()},
            drop_count=int(raw["drop_count"]),
            dropped_boxes=int(raw["dropped_boxes"]),
            skipped=tuple(str(s) for s in raw["skipped"]),
            errors=tuple(str(s) for s in raw["errors"]),
        )
