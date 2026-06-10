"""End-to-end T9 tuning: detections -> simulation -> sweep -> artifacts.

``tune_thresholds`` is the single entry the CLI (models/thresholds.py) and
the manager notebook call. Determinism: identical inputs (JSONLs, truth,
report, config) produce byte-identical thresholds.yaml and sweep.csv --
sampling is seeded per object from cfg.seed.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from yolo_waste_sorter.models.evaluation.report import EvalReport
from yolo_waste_sorter.models.thresholding.artifacts import (
    SWEEP_FILENAME,
    THRESHOLDS_FILENAME,
    write_sweep_csv,
    write_thresholds_yaml,
)
from yolo_waste_sorter.models.thresholding.consensus import ThresholdError, ThresholdParams
from yolo_waste_sorter.models.thresholding.simulate import (
    build_sightings,
    check_universe,
    load_detections,
)
from yolo_waste_sorter.models.thresholding.tuner import (
    SweepCell,
    per_class_tau,
    select_cell,
    sweep_cells,
)
from yolo_waste_sorter.utils.config import Config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TuneResult:
    """Selected rule, its simulated metrics, and the emitted artifact paths."""

    params: ThresholdParams
    constraint_met: bool
    wrong_bin_rate: float
    rest_rate: float
    cells: tuple[SweepCell, ...]
    chosen_index: int
    thresholds_path: Path
    sweep_path: Path


def tune_thresholds(
    cfg: Config,
    known_jsonl: Path,
    truth: Mapping[str, int],
    out_dir: Path,
    *,
    wilderness_jsonl: Path | None = None,
    report: EvalReport | None = None,
) -> TuneResult:
    """Sweep the T9 consensus rule and emit thresholds.yaml + sweep.csv.

    Args:
        cfg: Typed experiment config (seed, thresholds grids, conf_floor).
        known_jsonl: Evaluation detections dump over the VAL split
            (clean + degraded copies); predicted class_id/score per frame.
        truth: ``object_id -> true class_id`` for ALL val objects (build it
            with ``simulate.truth_from_manifest``); objects with no
            detections at all simulate as zero votes -> REST.
        out_dir: Artifact destination directory.
        wilderness_jsonl: Optional unknown-object probe (census drop list,
            T9/F13): every object in it is out-of-distribution, so any bin
            assignment counts as wrong-bin and REST is correct.
        report: Evaluation report; when its VAL conf_at_p95 spans more than
            0.1 across classes the sweep runs in per-class tau mode (F12).
            None keeps a global tau (e.g. before the first eval run).
    """
    known_dets = load_detections(known_jsonl)
    check_universe(known_dets, truth, where=str(known_jsonl))
    known = build_sightings(known_dets, sorted(truth), cfg.seed)

    wilderness: dict[str, list[tuple[int, float]]] = {}
    if wilderness_jsonl is not None:
        wild_dets = load_detections(wilderness_jsonl)
        wild_ids = sorted({d.object_id for d in wild_dets})
        overlap = sorted(set(wild_ids) & set(truth))
        if overlap:
            raise ThresholdError(
                f"wilderness object_id(s) collide with val objects: {overlap[:5]} -- "
                "the probe set must be disjoint from the known dataset"
            )
        wilderness = build_sightings(wild_dets, wild_ids, cfg.seed)
    else:
        logger.warning(
            "thresholds: tuning WITHOUT a wilderness probe -- wrong_bin_rate "
            "reflects known-class confusions only (unknown-object leakage untested, F13)"
        )

    anchors = None if report is None else per_class_tau(report, cfg.thresholds.sweep)
    cells = sweep_cells(
        cfg.thresholds.sweep, cfg.thresholds.conf_floor, known, truth, wilderness, anchors
    )
    chosen, constraint_met = select_cell(cells)
    cell = cells[chosen]

    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds_path = out_dir / THRESHOLDS_FILENAME
    sweep_path = out_dir / SWEEP_FILENAME
    write_thresholds_yaml(cell, cfg.thresholds.conf_floor, constraint_met, thresholds_path)
    write_sweep_csv(cells, chosen, sweep_path)
    logger.info(
        "thresholds: selected tau=%s min_votes=%d high_water=%.2f "
        "(wrong_bin=%.4f rest=%.4f constraint_met=%s) -> %s",
        cell.tau_frame,
        cell.min_votes,
        cell.high_water,
        cell.wrong_bin_rate,
        cell.rest_rate,
        constraint_met,
        thresholds_path,
    )
    return TuneResult(
        params=ThresholdParams(
            tau_frame=cell.tau_frame,
            min_votes=cell.min_votes,
            high_water=cell.high_water,
            conf_floor=cfg.thresholds.conf_floor,
        ),
        constraint_met=constraint_met,
        wrong_bin_rate=cell.wrong_bin_rate,
        rest_rate=cell.rest_rate,
        cells=tuple(cells),
        chosen_index=chosen,
        thresholds_path=thresholds_path,
        sweep_path=sweep_path,
    )
