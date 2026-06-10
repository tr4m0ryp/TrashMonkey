"""Artifact loading for the plot stage (task 014) -- parse, never compute.

Input contracts (fail fast on any deviation):
- evaluation report: ``models/evaluation/report.py`` ``EvalReport.write_yaml``
- curve arrays: ``models/evaluation/curves.py`` ``save_curves`` .npz layout
- threshold sweep: task 012's ``sweep.csv`` with EXACTLY the columns
  ``tau_frame,min_votes,high_water,wrong_bin_rate,rest_rate,chosen``
- split manifest: ``data/split.py`` ``SplitResult.write_manifest``; image keys
  are ``<class>/<source>__<name>`` (the dedup stage's ``Item.key`` convention)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import yaml

from yolo_waste_sorter.models.evaluation.report import EvalReport, load_report

_F64 = npt.NDArray[np.float64]

SWEEP_COLUMNS = ("tau_frame", "min_votes", "high_water", "wrong_bin_rate", "rest_rate", "chosen")
CURVE_ARRAYS = ("classes", "confidence", "precision", "recall", "f1")


class PlotError(Exception):
    """A plot-stage input artifact is missing or malformed."""


def resolve_report(report_or_path: EvalReport | Path) -> EvalReport:
    """Accept a parsed ``EvalReport`` or the path of its YAML dump."""
    if isinstance(report_or_path, EvalReport):
        return report_or_path
    return load_report(report_or_path)


@dataclass(frozen=True)
class CurveArrays:
    """One tier's per-class confidence curves, as persisted by ``save_curves``."""

    classes: tuple[str, ...]  # row order for the 2-D arrays
    confidence: _F64  # (n_points,)
    precision: _F64  # (n_classes, n_points)
    recall: _F64  # (n_classes, n_points)
    f1: _F64  # (n_classes, n_points)


def load_curves_npz(path: Path) -> CurveArrays:
    """Load and shape-check a ``save_curves`` .npz file."""
    if not path.is_file():
        raise PlotError(f"curves file not found: {path}")
    with np.load(path) as data:
        missing = sorted(set(CURVE_ARRAYS) - set(data.files))
        if missing:
            raise PlotError(f"{path}: missing array(s): {', '.join(missing)}")
        classes = tuple(str(name) for name in data["classes"])
        confidence = np.asarray(data["confidence"], dtype=np.float64)
        curves = {
            label: np.asarray(data[label], dtype=np.float64)
            for label in ("precision", "recall", "f1")
        }
    expected = (len(classes), confidence.shape[0])
    for label, curve in curves.items():
        if curve.ndim != 2 or curve.shape != expected:
            raise PlotError(f"{path}: {label} shape {curve.shape} does not match {expected}")
    return CurveArrays(classes=classes, confidence=confidence, **curves)


@dataclass(frozen=True)
class SweepRow:
    """One consensus-rule grid cell from task 012's sweep.csv."""

    tau_frame: float
    min_votes: int
    high_water: float
    wrong_bin_rate: float
    rest_rate: float
    chosen: bool


def read_sweep_csv(path: Path) -> tuple[SweepRow, ...]:
    """Parse sweep.csv against the fixed task-012 contract."""
    if not path.is_file():
        raise PlotError(f"sweep file not found: {path}")
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != SWEEP_COLUMNS:
            raise PlotError(
                f"{path}: columns {reader.fieldnames} != contract {list(SWEEP_COLUMNS)}"
            )
        rows: list[SweepRow] = []
        for line_no, record in enumerate(reader, start=2):
            try:
                chosen = int(record["chosen"])
                if chosen not in (0, 1):
                    raise ValueError(f"chosen must be 0/1, got {chosen}")
                rows.append(
                    SweepRow(
                        tau_frame=float(record["tau_frame"]),
                        min_votes=int(record["min_votes"]),
                        high_water=float(record["high_water"]),
                        wrong_bin_rate=float(record["wrong_bin_rate"]),
                        rest_rate=float(record["rest_rate"]),
                        chosen=bool(chosen),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise PlotError(f"{path}:{line_no}: malformed sweep row: {exc}") from exc
    if not rows:
        raise PlotError(f"{path}: sweep.csv has no data rows")
    return tuple(rows)


def read_split_composition(path: Path) -> dict[str, dict[str, dict[str, int]]]:
    """split -> class -> source -> image count, from a split-stage manifest.

    The manifest stores per-split-per-class counts only; the per-source axis
    is recovered from the assignment keys (``<class>/<source>__<name>``).
    """
    if not path.is_file():
        raise PlotError(f"split manifest not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict) or raw.get("stage") != "split":
        raise PlotError(f"{path}: not a split-stage manifest (stage={raw.get('stage')!r})")
    assignments = raw.get("assignments")
    if not isinstance(assignments, dict) or not assignments:
        raise PlotError(f"{path}: missing or empty 'assignments' mapping")
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for key, split in assignments.items():
        key, split = str(key), str(split)
        class_name, _, filename = key.partition("/")
        source, sep, _ = filename.partition("__")
        if not class_name or not sep or not source:
            raise PlotError(f"{path}: assignment key {key!r} is not '<class>/<source>__<name>'")
        per_source = counts.setdefault(split, {}).setdefault(class_name, {})
        per_source[source] = per_source.get(source, 0) + 1
    return counts
