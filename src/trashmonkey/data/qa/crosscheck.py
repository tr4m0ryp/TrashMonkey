"""IoU cross-check of our auto-boxes against a human-labeled reference set."""

from dataclasses import dataclass
from pathlib import Path
from statistics import mean, quantiles
from typing import Any

from .boxes import iou_cxcywh, parse_label_file
from .report import TARGET_MEDIAN_IOU


@dataclass(frozen=True)
class IoUStats:
    """Distribution of per-image IoU between two YOLO label dirs."""

    n_ours: int
    n_reference: int
    n_paired: int
    per_image: dict[str, float]
    mean: float
    median: float
    q1: float
    q3: float
    min: float
    max: float
    frac_geq_target: float
    target: float = TARGET_MEDIAN_IOU

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_ours": self.n_ours,
            "n_reference": self.n_reference,
            "n_paired": self.n_paired,
            "mean": self.mean,
            "median": self.median,
            "q1": self.q1,
            "q3": self.q3,
            "min": self.min,
            "max": self.max,
            "frac_geq_target": self.frac_geq_target,
            "target": self.target,
        }


def _best_iou(ours_path: Path, reference_path: Path) -> float:
    """Best-match IoU between the boxes of one image pair (class-agnostic)."""
    ours = parse_label_file(ours_path)
    reference = parse_label_file(reference_path)
    if not ours or not reference:
        return 0.0  # a missing box on either side is a localization failure
    return max(iou_cxcywh(a, b) for a in ours for b in reference)


def iou_crosscheck(ours_dir: Path, reference_dir: Path) -> IoUStats:
    """Pair YOLO txt files by filename stem and score the IoU distribution.

    The reference set is human-labeled ground truth (e.g. Polygence TrashNet);
    the T3 acceptance bar is median IoU >= TARGET_MEDIAN_IOU.
    """
    ours = {p.stem: p for p in ours_dir.glob("*.txt")}
    reference = {p.stem: p for p in reference_dir.glob("*.txt")}
    common = sorted(ours.keys() & reference.keys())
    if not common:
        raise ValueError(f"no filename stems shared between {ours_dir} and {reference_dir}")

    per_image = {stem: _best_iou(ours[stem], reference[stem]) for stem in common}
    values = sorted(per_image.values())
    if len(values) >= 2:
        q1, med, q3 = quantiles(values, n=4, method="inclusive")
    else:
        q1 = med = q3 = values[0]
    return IoUStats(
        n_ours=len(ours),
        n_reference=len(reference),
        n_paired=len(common),
        per_image=per_image,
        mean=mean(values),
        median=med,
        q1=q1,
        q3=q3,
        min=values[0],
        max=values[-1],
        frac_geq_target=sum(v >= TARGET_MEDIAN_IOU for v in values) / len(values),
    )
