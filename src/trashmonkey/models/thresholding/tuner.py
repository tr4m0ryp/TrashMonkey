"""T9 grid sweep, tradeoff metrics, and Pareto-knee selection.

Sweeps cfg.thresholds.sweep (tau_frame x min_votes x high_water) over the
simulated consensus decisions, computes the wrong-bin vs rest-rate tradeoff,
and picks the Pareto knee subject to wrong_bin_rate <= MAX_WRONG_BIN (2%,
T9). The artifact writers live in ``artifacts.py``; the end-to-end entry is
``run.tune_thresholds``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from trashmonkey.models.evaluation.report import EvalReport
from trashmonkey.models.thresholding.consensus import (
    REST,
    ThresholdError,
    ThresholdParams,
    Vote,
    consensus_decision,
)
from trashmonkey.utils.config import SweepConfig

logger = logging.getLogger(__name__)

MAX_WRONG_BIN = 0.02  # T9: contamination ceiling the knee is selected under
PER_CLASS_SPAN = 0.1  # F12: per-class tau triggers when conf_at_p95 spans more


@dataclass(frozen=True)
class SweepCell:
    """One grid cell: its rule parameters and simulated tradeoff metrics."""

    tau_frame: float | dict[int, float]
    tau_mean: float  # == tau_frame when global; mean of per-class taus otherwise
    min_votes: int
    high_water: float
    wrong_bin_rate: float
    rest_rate: float


def per_class_tau(report: EvalReport, sweep: SweepConfig) -> dict[int, float] | None:
    """Per-class tau anchors from the VAL conf_at_p95 curve, or None for global.

    F12/T9 criterion: per-class thresholds are worth it iff the smallest
    confidence sustaining precision >= 0.95 spans more than 0.1 across
    classes on the VAL curves. A class whose floor is never sustained
    (conf_at_p95 None) anchors at the sweep maximum -- the most conservative
    qualified-vote bar. Anchors are clamped into the sweep range; the sweep
    then shifts them so their mean tracks each tau_frame grid value.
    """
    lo, hi = min(sweep.tau_frame), max(sweep.tau_frame)
    anchors: dict[int, float] = {}
    for class_id, name in enumerate(report.classes):
        block = report.val.per_class.get(name)
        if block is None:
            raise ThresholdError(
                f"class {name!r} missing from the VAL per-class report -- "
                "cannot anchor its tau_frame"
            )
        raw = hi if block.conf_at_p95 is None else block.conf_at_p95
        anchors[class_id] = min(max(raw, lo), hi)
    span = max(anchors.values()) - min(anchors.values())
    if span <= PER_CLASS_SPAN:
        logger.info("thresholds: conf_at_p95 span %.3f <= %.1f -- global tau", span, PER_CLASS_SPAN)
        return None
    logger.info("thresholds: conf_at_p95 span %.3f > %.1f -- per-class tau", span, PER_CLASS_SPAN)
    return anchors


def _cell_tau(
    grid_tau: float, anchors: dict[int, float] | None, sweep: SweepConfig
) -> tuple[float | dict[int, float], float]:
    """A cell's tau (global float or shifted per-class mapping) and its mean."""
    if anchors is None:
        return grid_tau, grid_tau
    lo, hi = min(sweep.tau_frame), max(sweep.tau_frame)
    center = sum(anchors.values()) / len(anchors)
    taus = {
        class_id: min(max(anchor + grid_tau - center, lo), hi)
        for class_id, anchor in anchors.items()
    }
    return taus, sum(taus.values()) / len(taus)


def evaluate_cell(
    params: ThresholdParams,
    known: Mapping[str, Sequence[Vote]],
    truth: Mapping[str, int],
    wilderness: Mapping[str, Sequence[Vote]],
) -> tuple[float, float]:
    """(wrong_bin_rate, rest_rate) of one cell over the simulated objects.

    Wrong bin: a known object sorted to a class != its truth, or a
    wilderness (unknown) object sorted to ANY bin; REST is always correct
    for wilderness. Rates: wrong_bin over ALL objects, rest over KNOWN only.
    """
    wrong = 0
    rested = 0
    for object_id, votes in known.items():
        decision = consensus_decision(votes, params)
        if decision is REST:
            rested += 1
        elif decision != truth[object_id]:
            wrong += 1
    for votes in wilderness.values():
        if consensus_decision(votes, params) is not REST:
            wrong += 1
    total = len(known) + len(wilderness)
    if not known:
        raise ThresholdError("no known objects to simulate -- check the detections inputs")
    return wrong / total, rested / len(known)


def sweep_cells(
    sweep: SweepConfig,
    conf_floor: float,
    known: Mapping[str, Sequence[Vote]],
    truth: Mapping[str, int],
    wilderness: Mapping[str, Sequence[Vote]],
    anchors: dict[int, float] | None,
) -> list[SweepCell]:
    """Evaluate every (tau_frame, min_votes, high_water) grid cell, in grid order."""
    if not (sweep.tau_frame and sweep.min_votes and sweep.high_water):
        raise ThresholdError("cfg.thresholds.sweep: every grid must be non-empty")
    cells: list[SweepCell] = []
    for grid_tau in sweep.tau_frame:
        tau, tau_mean = _cell_tau(grid_tau, anchors, sweep)
        for min_votes in sweep.min_votes:
            for high_water in sweep.high_water:
                params = ThresholdParams(
                    tau_frame=tau,
                    min_votes=min_votes,
                    high_water=high_water,
                    conf_floor=conf_floor,
                )
                wrong_bin_rate, rest_rate = evaluate_cell(params, known, truth, wilderness)
                cells.append(
                    SweepCell(
                        tau_frame=tau,
                        tau_mean=tau_mean,
                        min_votes=min_votes,
                        high_water=high_water,
                        wrong_bin_rate=wrong_bin_rate,
                        rest_rate=rest_rate,
                    )
                )
    return cells


def pareto_front(cells: Sequence[SweepCell]) -> list[int]:
    """Indices of cells not dominated on (wrong_bin_rate, rest_rate)."""
    front: list[int] = []
    for i, cell in enumerate(cells):
        dominated = any(
            other.wrong_bin_rate <= cell.wrong_bin_rate
            and other.rest_rate <= cell.rest_rate
            and (other.wrong_bin_rate < cell.wrong_bin_rate or other.rest_rate < cell.rest_rate)
            for other in cells
        )
        if not dominated:
            front.append(i)
    return front


def select_cell(cells: Sequence[SweepCell]) -> tuple[int, bool]:
    """Pareto knee subject to wrong_bin <= MAX_WRONG_BIN; loud fallback otherwise.

    Among Pareto-front cells meeting the constraint, pick the lowest
    rest_rate (ties: lower wrong_bin, then grid order). When NO cell meets
    it, fall back to the overall lowest wrong_bin (ties: lower rest_rate)
    and flag constraint_met False.
    """
    front = pareto_front(cells)
    feasible = [i for i in front if cells[i].wrong_bin_rate <= MAX_WRONG_BIN]
    if feasible:
        chosen = min(feasible, key=lambda i: (cells[i].rest_rate, cells[i].wrong_bin_rate, i))
        return chosen, True
    chosen = min(range(len(cells)), key=lambda i: (cells[i].wrong_bin_rate, cells[i].rest_rate, i))
    logger.error(
        "thresholds: NO sweep cell meets wrong_bin_rate <= %.2f (best %.4f); "
        "falling back to the lowest-wrong-bin cell -- constraint_met: false",
        MAX_WRONG_BIN,
        cells[chosen].wrong_bin_rate,
    )
    return chosen, False
