"""Confidence-curve extraction from the ultralytics DetMetrics surface.

Verified against ultralytics 8.3.253 ``utils/metrics.py``: ``DetMetrics
.curves_results`` delegates to ``Metric.curves_results``, a list of four
``[x, y, xlabel, ylabel]`` entries -- the Confidence/Precision entry carries
``px`` (1000-point linspace over [0, 1]) and ``p_curve`` with one row per
class in ``box.ap_class_index`` order (``ap_per_class`` builds both). The
entries are located by their axis labels, never by position.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from trashmonkey.models.evaluation.report import EvalError

PRECISION_FLOOR = 0.95  # T9: per-class threshold anchor (precision >= 0.95)

_F64 = npt.NDArray[np.float64]


@dataclass(frozen=True)
class CurveSet:
    """Per-class metric-vs-confidence curves for one evaluation tier."""

    classes: tuple[str, ...]  # row order for the 2-D arrays
    confidence: _F64  # (n_points,)
    precision: _F64  # (n_classes, n_points)
    recall: _F64  # (n_classes, n_points)
    f1: _F64  # (n_classes, n_points)


def _curve_by_labels(results: Any, xlabel: str, ylabel: str) -> tuple[_F64, _F64]:
    """Locate one ``[x, y, xlabel, ylabel]`` entry of ``curves_results``."""
    for entry in results.curves_results:
        if len(entry) == 4 and str(entry[2]) == xlabel and str(entry[3]) == ylabel:
            return (
                np.asarray(entry[0], dtype=np.float64),
                np.asarray(entry[1], dtype=np.float64),
            )
    raise EvalError(
        f"no {xlabel!r}/{ylabel!r} curve in curves_results -- "
        "expected the ultralytics 8.3.x DetMetrics surface"
    )


def extract_curves(results: Any) -> CurveSet:
    """Pull the per-class confidence curves out of a DetMetrics-like object."""
    names = {int(class_id): str(name) for class_id, name in results.names.items()}
    classes = tuple(names[int(class_id)] for class_id in results.box.ap_class_index)
    confidence, precision = _curve_by_labels(results, "Confidence", "Precision")
    _, recall = _curve_by_labels(results, "Confidence", "Recall")
    _, f1 = _curve_by_labels(results, "Confidence", "F1")
    for label, curve in (("precision", precision), ("recall", recall), ("f1", f1)):
        if curve.ndim != 2 or curve.shape != (len(classes), confidence.shape[0]):
            raise EvalError(
                f"{label} curve shape {curve.shape} does not match "
                f"({len(classes)} classes, {confidence.shape[0]} points)"
            )
    return CurveSet(
        classes=classes, confidence=confidence, precision=precision, recall=recall, f1=f1
    )


def conf_at_precision(curves: CurveSet, floor: float = PRECISION_FLOOR) -> dict[str, float | None]:
    """Per class: smallest confidence from which precision STAYS >= ``floor``.

    Precision-vs-confidence is not monotone, so the first crossing can be a
    spike that dips again; the tuner needs the start of the final sustained
    region. None when the floor is never sustained (no usable threshold).
    """
    out: dict[str, float | None] = {}
    for row_index, name in enumerate(curves.classes):
        below = np.flatnonzero(curves.precision[row_index] < floor)
        if below.size == 0:
            out[name] = float(curves.confidence[0])
        elif int(below[-1]) + 1 >= curves.confidence.shape[0]:
            out[name] = None
        else:
            out[name] = float(curves.confidence[int(below[-1]) + 1])
    return out


def save_curves(curves: CurveSet, path: Path) -> Path:
    """Persist a tier's curves as .npz for the plot stage (014)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        classes=np.asarray(curves.classes),
        confidence=curves.confidence,
        precision=curves.precision,
        recall=curves.recall,
        f1=curves.f1,
    )
    return path
