"""TEST-2 severity-degradation curve: mAP vs ESP32 degradation level.

Severity 0 is the clean TEST-1 baseline (``report.test1``); severities 1..5
come from the report's TEST-2 ``SeverityPoint`` tuple. Render-only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from trashmonkey.models.evaluation.report import CLEAN_SEVERITY, EvalReport
from trashmonkey.visualization.loaders import resolve_report
from trashmonkey.visualization.style import finalize, setup_style


def plot_severity_curve(
    report_or_path: EvalReport | Path,
    save_path: Path | None = None,
) -> None:
    """mAP@50 and mAP@50-95 vs severity 0..5 (0 = clean TEST-1 baseline)."""
    setup_style()
    report = resolve_report(report_or_path)
    points: dict[int, tuple[float, float]] = {
        CLEAN_SEVERITY: (report.test1.map50, report.test1.map50_95)
    }
    for point in report.severity_curve:
        points[point.severity] = (point.map50, point.map50_95)
    severities = sorted(points)
    map50 = [points[s][0] for s in severities]
    map50_95 = [points[s][1] for s in severities]

    colors = sns.color_palette("colorblind", n_colors=2)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot(severities, map50, marker="o", ms=5, lw=1.4, color=colors[0], label="mAP@50")
    ax.plot(severities, map50_95, marker="s", ms=5, lw=1.4, color=colors[1], label="mAP@50-95")
    ax.set_xticks(severities)
    ax.set_xlabel("Degradation severity (0 = clean TEST-1)")
    ax.set_ylabel("mAP")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("TEST-2 robustness to camera degradation")
    ax.legend(loc="lower left", frameon=False)
    finalize(fig, save_path)
