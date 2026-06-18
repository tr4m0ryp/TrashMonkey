"""Tests for post-hoc temperature-scaling calibration (task 002). Pure, no GPU.

Synthetic data is seeded with ``numpy.random.default_rng(42)`` so the
over-confident set and its calibration outcome are reproducible.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from trashmonkey.models.thresholding import (
    CalibrationError,
    Calibrator,
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
)

_F64 = npt.NDArray[np.float64]
SEED = 42


def _overconfident_set(n: int = 6000) -> tuple[_F64, _F64]:
    """Synthetic detector that is systematically over-confident.

    Latent probability ``p`` decides correctness (so ``p`` is the *true*
    calibrated probability), but the reported score steepens the logit by a
    factor > 1 -- a genuine temperature miscalibration that pushes scores
    toward the extremes, making the detector claim more confidence than its
    accuracy warrants. Temperature scaling with the right ``T > 1`` should
    soften this back toward the diagonal.
    """
    rng = np.random.default_rng(SEED)
    p = rng.uniform(0.02, 0.98, size=n)
    labels = (rng.uniform(0.0, 1.0, size=n) < p).astype(np.float64)
    logits = np.log(p) - np.log1p(-p)
    scores = np.clip(1.0 / (1.0 + np.exp(-(logits * 2.5))), 1e-6, 1.0 - 1e-6)
    return scores, labels


def test_apply_temperature_identity_at_one() -> None:
    for s in (0.01, 0.2, 0.5, 0.73, 0.99):
        assert apply_temperature(s, 1.0) == pytest.approx(s, abs=1e-9)


def test_apply_temperature_monotonic_in_score() -> None:
    grid = np.linspace(0.0, 1.0, 200)
    for temperature in (0.3, 1.0, 2.5, 10.0):
        calibrated = [apply_temperature(float(s), temperature) for s in grid]
        diffs = np.diff(calibrated)
        assert np.all(diffs >= -1e-12), f"not monotonic at T={temperature}"
        # Strictly increasing on the interior (no flat plateau).
        assert np.all(np.diff(calibrated[1:-1]) > 0.0)


def test_apply_temperature_softens_above_one_sharpens_below() -> None:
    # A confident score (>0.5): T>1 pulls it down, T<1 pushes it up.
    assert apply_temperature(0.9, 3.0) < 0.9
    assert apply_temperature(0.9, 0.5) > 0.9
    # A timid score (<0.5): T>1 pulls it up toward 0.5, T<1 pushes it down.
    assert apply_temperature(0.1, 3.0) > 0.1
    assert apply_temperature(0.1, 0.5) < 0.1
    # 0.5 is the logit-zero fixed point for every temperature.
    assert apply_temperature(0.5, 7.0) == pytest.approx(0.5, abs=1e-9)


def test_apply_temperature_clamps_saturated_scores() -> None:
    # Exact 0 and 1 must not produce inf/nan via the logit.
    for s in (0.0, 1.0):
        out = apply_temperature(s, 2.0)
        assert np.isfinite(out)
        assert 0.0 < out < 1.0


def test_apply_temperature_rejects_bad_temperature() -> None:
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(CalibrationError):
            apply_temperature(0.5, bad)


def test_fit_recovers_known_temperature() -> None:
    # Build labels from a known calibrated p, then over-sharpen the scores by
    # a known T_true; the fit should recover ~T_true.
    rng = np.random.default_rng(SEED)
    n = 6000
    p_true = rng.uniform(0.02, 0.98, size=n)
    labels = (rng.uniform(0.0, 1.0, size=n) < p_true).astype(np.float64)
    t_true = 2.5
    logits = np.log(p_true) - np.log1p(-p_true)
    # Over-confident: steepen the logits (multiply), so recovering calibration
    # needs T_true > 1. ``fit`` should divide by ~T_true to undo it.
    scores = 1.0 / (1.0 + np.exp(-(logits * t_true)))
    t_hat = fit_temperature(scores, labels)
    assert t_hat == pytest.approx(t_true, rel=0.15)


def test_calibration_reduces_ece_and_nll() -> None:
    scores, labels = _overconfident_set()
    ece_before = expected_calibration_error(scores, labels)
    nll_before = negative_log_likelihood(scores, labels, 1.0)

    t = fit_temperature(scores, labels)
    assert t > 1.0  # an over-confident detector needs softening

    cal = Calibrator(t)
    calibrated = cal.calibrate_all(scores)
    ece_after = expected_calibration_error(calibrated, labels)
    nll_after = negative_log_likelihood(scores, labels, t)

    assert ece_after < ece_before
    assert nll_after < nll_before
    # The improvement is substantial, not marginal noise.
    assert ece_after < 0.5 * ece_before


def test_negative_log_likelihood_minimised_at_fit() -> None:
    scores, labels = _overconfident_set()
    t = fit_temperature(scores, labels)
    best = negative_log_likelihood(scores, labels, t)
    for other in (0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0):
        assert best <= negative_log_likelihood(scores, labels, other) + 1e-9


def test_calibrator_matches_apply_temperature() -> None:
    cal = Calibrator(2.3)
    for s in (0.05, 0.4, 0.88):
        assert cal.calibrate(s) == pytest.approx(apply_temperature(s, 2.3), abs=1e-12)
    arr = cal.calibrate_all(np.array([0.05, 0.4, 0.88]))
    expected = [apply_temperature(s, 2.3) for s in (0.05, 0.4, 0.88)]
    assert np.allclose(arr, expected, atol=1e-12)


def test_calibrator_rejects_nonpositive_temperature() -> None:
    for bad in (0.0, -2.0, float("nan")):
        with pytest.raises(CalibrationError):
            Calibrator(bad)


def test_ece_perfect_calibration_is_low() -> None:
    # Scores that equal the true probability bin into matching accuracy.
    rng = np.random.default_rng(SEED)
    n = 20000
    p = rng.uniform(0.0, 1.0, size=n)
    labels = (rng.uniform(0.0, 1.0, size=n) < p).astype(np.float64)
    assert expected_calibration_error(p, labels) < 0.02


def test_fit_returns_near_one_for_calibrated_input() -> None:
    rng = np.random.default_rng(SEED)
    n = 8000
    p = rng.uniform(0.02, 0.98, size=n)
    labels = (rng.uniform(0.0, 1.0, size=n) < p).astype(np.float64)
    t = fit_temperature(p, labels)
    assert t == pytest.approx(1.0, abs=0.2)


def test_input_validation() -> None:
    with pytest.raises(CalibrationError):
        fit_temperature(np.array([0.5, 0.6]), np.array([1.0]))  # shape mismatch
    with pytest.raises(CalibrationError):
        fit_temperature(np.array([]), np.array([]))  # empty
    with pytest.raises(CalibrationError):
        fit_temperature(np.array([1.5]), np.array([1.0]))  # score out of [0, 1]
    with pytest.raises(CalibrationError):
        fit_temperature(np.array([0.5]), np.array([0.5]))  # non-binary label
    with pytest.raises(CalibrationError):
        expected_calibration_error(np.array([0.5]), np.array([1.0]), n_bins=0)
