"""QA report types, provenance records, and acceptance constants (T3 QA plan)."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Acceptance bars from the T3 decision (research/yolo11-waste-detection-finetune.md).
REVIEW_FAIL_MAX = 0.10  # <=10% of human-reviewed sample boxes may fail overall
LOC_FAIL_MAX = 0.20  # <=20% may fail on localization-only grounds
TARGET_MEDIAN_IOU = 0.80  # vs human-annotated reference set (Polygence TrashNet)

VALID_METHODS = frozenset({"dino", "birefnet", "centerbox"})


@dataclass(frozen=True)
class ProvenanceRecord:
    """One autobox provenance JSONL line."""

    image: str
    source: str
    method: str
    confidence: float
    flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.method not in VALID_METHODS:
            raise ValueError(f"unknown autobox method {self.method!r} for {self.image!r}")

    @property
    def stem(self) -> str:
        return Path(self.image).stem


def load_provenance(path: Path) -> dict[str, ProvenanceRecord]:
    """Read autobox provenance JSONL, keyed by image filename stem."""
    records: dict[str, ProvenanceRecord] = {}
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON") from exc
        record = ProvenanceRecord(
            image=obj["image"],
            source=obj["source"],
            method=obj["method"],
            confidence=float(obj["confidence"]),
            flags=tuple(obj.get("flags", [])),
        )
        if record.stem in records:
            raise ValueError(f"{path}:{lineno}: duplicate provenance for stem {record.stem!r}")
        records[record.stem] = record
    return records


@dataclass
class ImageQA:
    """Per-image QA outcome: provenance context plus flags raised by checks."""

    image: str
    stem: str
    source: str
    method: str
    confidence: float
    n_boxes: int
    class_id: int | None
    flags: list[str] = field(default_factory=list)


@dataclass
class QAReport:
    """Result of run_checks; later enriched with review + crosscheck metrics."""

    labels_dir: str
    images: dict[str, ImageQA]
    review_fail_rate: float | None = None
    loc_fail_rate: float | None = None
    median_iou: float | None = None

    @property
    def total_images(self) -> int:
        return len(self.images)

    @property
    def flagged(self) -> dict[str, ImageQA]:
        return {stem: rec for stem, rec in self.images.items() if rec.flags}

    @property
    def flag_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rec in self.images.values():
            for flag in rec.flags:
                counts[flag] = counts.get(flag, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def acceptance_pass(self) -> bool:
        """True iff all three T3 acceptance bars are met. Raises if metrics unset."""
        if self.review_fail_rate is None or self.loc_fail_rate is None or self.median_iou is None:
            raise ValueError(
                "acceptance metrics incomplete: set review_fail_rate, loc_fail_rate "
                "and median_iou before reading acceptance_pass"
            )
        return (
            self.review_fail_rate <= REVIEW_FAIL_MAX
            and self.loc_fail_rate <= LOC_FAIL_MAX
            and self.median_iou >= TARGET_MEDIAN_IOU
        )

    def to_dict(self) -> dict[str, Any]:
        metrics_set = None not in (self.review_fail_rate, self.loc_fail_rate, self.median_iou)
        return {
            "labels_dir": self.labels_dir,
            "total_images": self.total_images,
            "flagged_images": len(self.flagged),
            "flag_counts": self.flag_counts,
            "per_image_flags": {stem: list(rec.flags) for stem, rec in sorted(self.images.items())},
            "acceptance": {
                "review_fail_rate": self.review_fail_rate,
                "review_fail_max": REVIEW_FAIL_MAX,
                "loc_fail_rate": self.loc_fail_rate,
                "loc_fail_max": LOC_FAIL_MAX,
                "median_iou": self.median_iou,
                "target_median_iou": TARGET_MEDIAN_IOU,
                "pass": self.acceptance_pass if metrics_set else None,
            },
        }

    def to_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
