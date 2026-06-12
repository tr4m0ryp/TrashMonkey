"""Publication figures from evaluation/tuning artifacts (task 014).

Stable import surface: all rendering lives in the sibling modules; import
everything from here. Render-only -- no metric is computed in this package.
"""

from trashmonkey.visualization.composition import plot_dataset_composition
from trashmonkey.visualization.curves import plot_per_class_curves
from trashmonkey.visualization.degradation import plot_degradation_grid
from trashmonkey.visualization.loaders import (
    CURVE_ARRAYS,
    SWEEP_COLUMNS,
    CurveArrays,
    DetectionLine,
    PlotError,
    SweepRow,
    load_curves_npz,
    read_detections_jsonl,
    read_results_csv,
    read_split_composition,
    read_sweep_csv,
    resolve_report,
)
from trashmonkey.visualization.openset import (
    plot_confidence_separation,
    top_scores_per_frame,
)
from trashmonkey.visualization.severity import plot_severity_curve
from trashmonkey.visualization.style import (
    DPI,
    LABEL_SIZE,
    PALETTE,
    TICK_SIZE,
    finalize,
    series_colors,
    setup_style,
)
from trashmonkey.visualization.tiers import plot_tier_comparison
from trashmonkey.visualization.tradeoff import (
    MAX_WRONG_BIN,
    pareto_front,
    plot_threshold_tradeoff,
)
from trashmonkey.visualization.training import plot_training_curves

__all__ = [
    "CURVE_ARRAYS",
    "DPI",
    "LABEL_SIZE",
    "MAX_WRONG_BIN",
    "PALETTE",
    "SWEEP_COLUMNS",
    "TICK_SIZE",
    "CurveArrays",
    "DetectionLine",
    "PlotError",
    "SweepRow",
    "finalize",
    "load_curves_npz",
    "pareto_front",
    "plot_confidence_separation",
    "plot_dataset_composition",
    "plot_degradation_grid",
    "plot_per_class_curves",
    "plot_severity_curve",
    "plot_threshold_tradeoff",
    "plot_tier_comparison",
    "plot_training_curves",
    "read_detections_jsonl",
    "read_results_csv",
    "read_split_composition",
    "read_sweep_csv",
    "resolve_report",
    "series_colors",
    "setup_style",
    "top_scores_per_frame",
]
