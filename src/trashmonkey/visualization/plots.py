"""Publication figures from evaluation/tuning artifacts (task 014).

Stable import surface: all rendering lives in the sibling modules; import
everything from here. Render-only -- no metric is computed in this package.
"""

from trashmonkey.visualization.composition import plot_dataset_composition
from trashmonkey.visualization.curves import plot_per_class_curves
from trashmonkey.visualization.loaders import (
    CURVE_ARRAYS,
    SWEEP_COLUMNS,
    CurveArrays,
    PlotError,
    SweepRow,
    load_curves_npz,
    read_split_composition,
    read_sweep_csv,
    resolve_report,
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
from trashmonkey.visualization.tradeoff import (
    MAX_WRONG_BIN,
    pareto_front,
    plot_threshold_tradeoff,
)

__all__ = [
    "CURVE_ARRAYS",
    "DPI",
    "LABEL_SIZE",
    "MAX_WRONG_BIN",
    "PALETTE",
    "SWEEP_COLUMNS",
    "TICK_SIZE",
    "CurveArrays",
    "PlotError",
    "SweepRow",
    "finalize",
    "load_curves_npz",
    "pareto_front",
    "plot_dataset_composition",
    "plot_per_class_curves",
    "plot_severity_curve",
    "plot_threshold_tradeoff",
    "read_split_composition",
    "read_sweep_csv",
    "resolve_report",
    "series_colors",
    "setup_style",
]
