"""T9 rest-bin threshold tuning (task 012): consensus rule, sim, sweep.

The public surface is re-exported by ``yolo_waste_sorter.models.thresholds``
(the stable import path + CLI). The Jetson runtime (015) needs only
``consensus_decision``, ``REST``, and ``ThresholdParams``.
"""

from yolo_waste_sorter.models.thresholding.artifacts import (
    SWEEP_COLUMNS,
    SWEEP_FILENAME,
    THRESHOLDS_FILENAME,
    write_sweep_csv,
    write_thresholds_yaml,
)
from yolo_waste_sorter.models.thresholding.consensus import (
    REST,
    Decision,
    RestType,
    ThresholdError,
    ThresholdParams,
    Vote,
    consensus_decision,
)
from yolo_waste_sorter.models.thresholding.run import TuneResult, tune_thresholds
from yolo_waste_sorter.models.thresholding.simulate import (
    SIGHTINGS_RANGE,
    Detection,
    build_sightings,
    load_detections,
    sample_sightings,
    top_votes_per_frame,
    truth_from_manifest,
)
from yolo_waste_sorter.models.thresholding.tuner import (
    MAX_WRONG_BIN,
    PER_CLASS_SPAN,
    SweepCell,
    evaluate_cell,
    pareto_front,
    per_class_tau,
    select_cell,
    sweep_cells,
)

__all__ = [
    "MAX_WRONG_BIN",
    "PER_CLASS_SPAN",
    "REST",
    "SIGHTINGS_RANGE",
    "SWEEP_COLUMNS",
    "SWEEP_FILENAME",
    "THRESHOLDS_FILENAME",
    "Decision",
    "Detection",
    "RestType",
    "SweepCell",
    "ThresholdError",
    "ThresholdParams",
    "TuneResult",
    "Vote",
    "build_sightings",
    "consensus_decision",
    "evaluate_cell",
    "load_detections",
    "pareto_front",
    "per_class_tau",
    "sample_sightings",
    "select_cell",
    "sweep_cells",
    "top_votes_per_frame",
    "truth_from_manifest",
    "tune_thresholds",
    "write_sweep_csv",
    "write_thresholds_yaml",
]
