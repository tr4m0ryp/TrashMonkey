"""Per-class mAP@50 across the three evaluation tiers, as grouped bars.

The domain-gap figure: VAL (optimistic), TEST-1 (held-out source), and the
worst TEST-2 severity (degraded imaging) side by side per material class.
A class absent from a tier's ground truth renders as a zero-height bar.
Render-only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from trashmonkey.models.evaluation.report import EvalReport, TierReport
from trashmonkey.visualization.loaders import resolve_report
from trashmonkey.visualization.style import finalize, series_colors, setup_style


def _tier_series(report: EvalReport) -> list[tuple[str, TierReport]]:
    series = [("VAL", report.val), ("TEST-1", report.test1)]
    if report.test2:
        worst = max(report.test2, key=lambda tier: tier.severity)
        series.append((f"TEST-2 (s{worst.severity})", worst))
    return series


def plot_tier_comparison(
    report_or_path: EvalReport | Path,
    save_path: Path | None = None,
) -> None:
    """Grouped per-class mAP@50 bars: VAL vs TEST-1 vs worst TEST-2 severity."""
    setup_style()
    report = resolve_report(report_or_path)
    series = _tier_series(report)
    classes = [
        name
        for name in report.classes
        if any(name in tier.per_class for _, tier in series)
    ]
    labels = [f"{label} (overall {tier.map50:.2f})" for label, tier in series]
    colors = series_colors(labels)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    positions = np.arange(len(classes))
    width = 0.8 / len(series)
    for column, ((_, tier), label) in enumerate(zip(series, labels, strict=True)):
        entry = tier.per_class
        heights = [entry[name].map50 if name in entry else 0.0 for name in classes]
        offset = (column - (len(series) - 1) / 2) * width
        ax.bar(positions + offset, heights, width=width * 0.92, color=colors[label], label=label)
    ax.set_xticks(positions)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("mAP@50")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Per-class accuracy across evaluation tiers")
    ax.legend(loc="lower right", frameon=False)
    finalize(fig, save_path)
