"""Per-class confidence curves: P-vs-confidence panel + PR-style panel.

Renders the .npz arrays persisted by ``models/evaluation/curves.save_curves``.
The ``conf_at_p95`` markers come from the report tier whose ``curves_path``
matches the given .npz (falling back to VAL, the only tier thresholds may be
tuned on).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from trashmonkey.models.evaluation.curves import PRECISION_FLOOR
from trashmonkey.models.evaluation.report import EvalReport, TierReport
from trashmonkey.visualization.loaders import (
    CurveArrays,
    load_curves_npz,
    resolve_report,
)
from trashmonkey.visualization.style import finalize, series_colors, setup_style


def _tier_for_curves(report: EvalReport, curves_npz_path: Path) -> TierReport:
    """The tier that produced this .npz (by curves_path), else VAL."""
    for tier in (report.val, report.test1, *report.test2):
        if Path(tier.curves_path).name == curves_npz_path.name:
            return tier
    return report.val


def _draw_precision_panel(ax: Axes, curves: CurveArrays, tier: TierReport) -> None:
    colors = series_colors(curves.classes)
    for row, name in enumerate(curves.classes):
        ax.plot(curves.confidence, curves.precision[row], color=colors[name], lw=1.2, label=name)
        entry = tier.per_class.get(name)
        if entry is not None and entry.conf_at_p95 is not None:
            conf = entry.conf_at_p95
            prec = float(np.interp(conf, curves.confidence, curves.precision[row]))
            ax.plot(
                [conf], [prec], marker="o", ms=5, color=colors[name],
                mec="black", mew=0.6, zorder=5,
            )
    ax.axhline(PRECISION_FLOOR, color="grey", ls="--", lw=0.8)
    ax.annotate(
        f"P = {PRECISION_FLOOR:.2f}", xy=(0.02, PRECISION_FLOOR),
        xytext=(2, 3), textcoords="offset points", fontsize=8, color="grey",
    )
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Precision vs confidence ({tier.tier})")
    ax.legend(loc="lower right", frameon=False)


def _draw_pr_panel(ax: Axes, curves: CurveArrays, tier: TierReport) -> None:
    colors = series_colors(curves.classes)
    for row, name in enumerate(curves.classes):
        ax.plot(curves.recall[row], curves.precision[row], color=colors[name], lw=1.2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Precision vs recall ({tier.tier})")


def plot_per_class_curves(
    report_or_path: EvalReport | Path,
    curves_npz_path: Path,
    save_path: Path | None = None,
) -> None:
    """Per-class P-vs-confidence (with conf_at_p95 markers) + PR-style panel."""
    setup_style()
    report = resolve_report(report_or_path)
    curves = load_curves_npz(curves_npz_path)
    tier = _tier_for_curves(report, curves_npz_path)
    fig, (ax_conf, ax_pr) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    _draw_precision_panel(ax_conf, curves, tier)
    _draw_pr_panel(ax_pr, curves, tier)
    finalize(fig, save_path)
