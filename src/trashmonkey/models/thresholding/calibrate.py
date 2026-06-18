"""Post-hoc temperature-scaling confidence calibration (task 002).

Detector scores (``models/evaluation/detections.py`` -- the ``score`` field)
are post-sigmoid probabilities in [0, 1]. To make the rest-bin reject
thresholds map to *true* probabilities, we fit a single scalar temperature
``T`` to the held-out (score, correctness) pairs and rescale every score
through the logit space::

    logit = ln(p / (1 - p))
    p'    = sigmoid(logit / T)

``T = 1`` is the identity; ``T > 1`` softens (reduces) over-confident scores,
``T < 1`` sharpens them. The map is strictly monotonic in the score for any
``T > 0``, so it preserves ranking (and therefore precision/recall at any
operating point) while re-spacing the probabilities.

Pure numpy/python: no torch, no ultralytics. ``fit_temperature`` runs a
deterministic golden-section search over ``T`` minimising the negative
log-likelihood; the search has no RNG, so it is reproducible by construction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

_F64 = npt.NDArray[np.float64]

# Probabilities are clamped this far from {0, 1} before the logit so that
# saturated detector scores do not produce +-inf logits (and the NLL stays
# finite). 1e-6 keeps |logit| <= ~13.8, well inside float64 range.
_EPS = 1e-6

# Golden-section search bounds and tolerance for the 1-D temperature fit.
_T_LO = 1e-2
_T_HI = 1e2
_T_TOL = 1e-4
_GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0  # 1/phi ~= 0.618


class CalibrationError(ValueError):
    """Raised when the calibration inputs are degenerate or mismatched."""


@dataclass(frozen=True)
class Calibrator:
    """A fitted temperature wrapper. Apply with :meth:`calibrate`."""

    temperature: float

    def __post_init__(self) -> None:
        if not (self.temperature > 0.0) or not np.isfinite(self.temperature):
            raise CalibrationError(f"temperature must be finite and > 0, got {self.temperature!r}")

    def calibrate(self, score: float) -> float:
        """Calibrate a single post-sigmoid score through the fitted temperature."""
        return apply_temperature(score, self.temperature)

    def calibrate_all(self, scores: Sequence[float] | _F64) -> _F64:
        """Calibrate an array of scores (vectorised)."""
        p = _clamp_prob(np.asarray(scores, dtype=np.float64))
        return _sigmoid(_logit(p) / self.temperature)


def _clamp_prob(p: _F64) -> _F64:
    """Clamp probabilities into ``[_EPS, 1 - _EPS]`` to keep logits finite."""
    return np.asarray(np.clip(p, _EPS, 1.0 - _EPS), dtype=np.float64)


def _logit(p: _F64) -> _F64:
    """Inverse sigmoid: ``ln(p / (1 - p))``. Input must already be clamped."""
    return np.asarray(np.log(p) - np.log1p(-p), dtype=np.float64)


def _sigmoid(z: _F64) -> _F64:
    """Numerically stable logistic sigmoid over an array of logits."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return np.asarray(out, dtype=np.float64)


def apply_temperature(score: float, temperature: float) -> float:
    """Rescale one post-sigmoid score by ``temperature`` in logit space.

    Strictly increasing in ``score`` for any ``temperature > 0`` (the logit is
    increasing and dividing by a positive constant preserves order), so the
    calibrated score keeps the detector's ranking.
    """
    if not (temperature > 0.0) or not np.isfinite(temperature):
        raise CalibrationError(f"temperature must be finite and > 0, got {temperature!r}")
    p = float(np.clip(score, _EPS, 1.0 - _EPS))
    z = float(np.log(p) - np.log1p(-p))
    return float(_sigmoid(np.asarray([z / temperature], dtype=np.float64))[0])


def _validate_pairs(scores: _F64, labels: _F64) -> None:
    if scores.shape != labels.shape:
        raise CalibrationError(
            f"scores and labels must share a shape, got {scores.shape} vs {labels.shape}"
        )
    if scores.ndim != 1 or scores.size == 0:
        raise CalibrationError("scores and labels must be non-empty 1-D arrays")
    if not np.all(np.isfinite(scores)):
        raise CalibrationError("scores contain non-finite values")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise CalibrationError("scores must be post-sigmoid probabilities in [0, 1]")
    if not np.all((labels == 0.0) | (labels == 1.0)):
        raise CalibrationError("labels must be binary correctness flags (0 or 1)")


def negative_log_likelihood(scores: _F64, labels: _F64, temperature: float) -> float:
    """Mean binary NLL of the temperature-scaled scores against ``labels``."""
    p = _clamp_prob(_sigmoid(_logit(_clamp_prob(scores)) / temperature))
    nll = -(labels * np.log(p) + (1.0 - labels) * np.log1p(-p))
    return float(np.mean(nll))


def expected_calibration_error(scores: _F64, labels: _F64, *, n_bins: int = 15) -> float:
    """Expected calibration error: |confidence - accuracy| over equal-width bins.

    Bins the scores into ``n_bins`` equal-width buckets over [0, 1], then sums
    the per-bin |mean-confidence - empirical-accuracy| weighted by bin mass.
    Lower is better; a perfectly calibrated detector scores 0.
    """
    if n_bins < 1:
        raise CalibrationError(f"n_bins must be >= 1, got {n_bins}")
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    _validate_pairs(scores, labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Right-closed bins; the score 1.0 lands in the final bin, not out of range.
    bin_idx = np.clip(np.digitize(scores, edges[1:-1], right=False), 0, n_bins - 1)
    total = scores.size
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        confidence = float(np.mean(scores[mask]))
        accuracy = float(np.mean(labels[mask]))
        ece += (count / total) * abs(confidence - accuracy)
    return ece


def fit_temperature(
    scores: Sequence[float] | _F64,
    labels: Sequence[float] | _F64,
    *,
    bounds: tuple[float, float] = (_T_LO, _T_HI),
    tol: float = _T_TOL,
) -> float:
    """Fit the temperature minimising NLL over (score, correctness) pairs.

    ``scores`` are post-sigmoid detector confidences in [0, 1]; ``labels`` are
    binary correctness flags (1 = the detection's class matched the truth).
    The objective is convex in ``T`` for fixed logits, so a golden-section
    search over the positive interval ``bounds`` converges to the global
    minimiser. Returns ``T > 0``; ``T == 1`` means the scores were already
    calibrated. Fully deterministic (no RNG).
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    _validate_pairs(s, y)
    logits = _logit(_clamp_prob(s))

    def objective(temperature: float) -> float:
        p = _clamp_prob(_sigmoid(logits / temperature))
        nll = -(y * np.log(p) + (1.0 - y) * np.log1p(-p))
        return float(np.mean(nll))

    lo, hi = bounds
    if not (0.0 < lo < hi):
        raise CalibrationError(f"bounds must satisfy 0 < lo < hi, got {bounds!r}")
    # Golden-section search: shrink [lo, hi] keeping the minimiser bracketed.
    c = hi - _GOLDEN * (hi - lo)
    d = lo + _GOLDEN * (hi - lo)
    fc, fd = objective(c), objective(d)
    while (hi - lo) > tol:
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - _GOLDEN * (hi - lo)
            fc = objective(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + _GOLDEN * (hi - lo)
            fd = objective(d)
    return float((lo + hi) / 2.0)
