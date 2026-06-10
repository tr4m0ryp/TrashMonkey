"""Reader for the T9 deployment artifact ``thresholds.yaml``.

The thresholding package (012) only WRITES the artifact
(``models.thresholding.artifacts.write_thresholds_yaml``); this is the
matching fail-fast reader the Jetson runtime uses. The schema is exactly what
the writer emits: ``tau_frame`` (float, or mapping class_id -> float in
per-class mode), ``min_votes``, ``high_water``, ``conf_floor``, plus the
informational ``constraint_met`` and ``selected_metrics`` blocks. Unknown or
missing keys raise -- no silent fallbacks.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from yolo_waste_sorter.models.thresholds import ThresholdError, ThresholdParams

_REQUIRED = ("tau_frame", "min_votes", "high_water", "conf_floor")
_ALLOWED = (*_REQUIRED, "constraint_met", "selected_metrics")


def _as_float(value: object, where: str) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise ThresholdError(f"{where}: expected a number, got {type(value).__name__}")


def _tau_frame(value: object, where: str) -> float | dict[int, float]:
    if isinstance(value, dict):
        if not value:
            raise ThresholdError(f"{where}: per-class mapping must not be empty")
        out: dict[int, float] = {}
        for key, tau in value.items():
            if not isinstance(key, int) or isinstance(key, bool):
                raise ThresholdError(f"{where}: class ids must be ints, got {key!r}")
            out[key] = _as_float(tau, f"{where}[{key}]")
        return out
    return _as_float(value, where)


def load_threshold_params(path: Path) -> ThresholdParams:
    """Parse and validate ``thresholds.yaml`` into the shared ``ThresholdParams``."""
    if not path.is_file():
        raise ThresholdError(f"thresholds artifact not found: {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ThresholdError(f"{path}: top level must be a mapping")
    unknown = [k for k in raw if k not in _ALLOWED]
    if unknown:
        keys = ", ".join(repr(k) for k in unknown)
        raise ThresholdError(f"{path}: unknown key(s) {keys}; allowed: {', '.join(_ALLOWED)}")
    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        raise ThresholdError(f"{path}: missing required key(s): {', '.join(missing)}")
    min_votes = raw["min_votes"]
    if not isinstance(min_votes, int) or isinstance(min_votes, bool) or min_votes < 1:
        raise ThresholdError(f"{path}: min_votes must be an int >= 1, got {min_votes!r}")
    return ThresholdParams(
        tau_frame=_tau_frame(raw["tau_frame"], f"{path}: tau_frame"),
        min_votes=min_votes,
        high_water=_as_float(raw["high_water"], f"{path}: high_water"),
        conf_floor=_as_float(raw["conf_floor"], f"{path}: conf_floor"),
    )
