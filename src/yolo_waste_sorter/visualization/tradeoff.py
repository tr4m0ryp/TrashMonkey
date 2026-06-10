"""Threshold sweep tradeoff cloud: wrong-bin rate vs rest-bin rate.

Renders task 012's ``sweep.csv`` grid: every cell as a scatter point, the
Pareto front (both rates minimized) as a line, the chosen knee starred and
annotated, and the wrong-bin constraint as a vertical guide.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from yolo_waste_sorter.models.thresholding.tuner import MAX_WRONG_BIN
from yolo_waste_sorter.visualization.loaders import SweepRow, read_sweep_csv
from yolo_waste_sorter.visualization.style import finalize, setup_style

__all__ = ["MAX_WRONG_BIN", "pareto_front", "plot_threshold_tradeoff"]


def pareto_front(rows: tuple[SweepRow, ...]) -> tuple[SweepRow, ...]:
    """Non-dominated cells when minimizing both wrong_bin_rate and rest_rate."""
    front: list[SweepRow] = []
    best_rest = float("inf")
    for row in sorted(rows, key=lambda r: (r.wrong_bin_rate, r.rest_rate)):
        if row.rest_rate < best_rest:
            front.append(row)
            best_rest = row.rest_rate
    return tuple(front)


def plot_threshold_tradeoff(sweep_csv_path: Path, save_path: Path | None = None) -> None:
    """Scatter all sweep cells, line the Pareto front, star the chosen knee."""
    setup_style()
    rows = read_sweep_csv(sweep_csv_path)
    front = pareto_front(rows)
    colors = sns.color_palette("colorblind", n_colors=4)

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.scatter(
        [r.wrong_bin_rate for r in rows], [r.rest_rate for r in rows],
        s=16, alpha=0.55, color=colors[0], edgecolors="none", label="sweep cells",
    )
    ax.plot(
        [r.wrong_bin_rate for r in front], [r.rest_rate for r in front],
        color=colors[1], lw=1.4, marker=".", ms=6, label="Pareto front",
    )
    ax.axvline(MAX_WRONG_BIN, color="grey", ls="--", lw=1.0)
    ax.annotate(
        f"wrong-bin cap {MAX_WRONG_BIN:.2f}", xy=(MAX_WRONG_BIN, 1.0),
        xycoords=("data", "axes fraction"), xytext=(3, -10),
        textcoords="offset points", fontsize=8, color="grey",
    )
    chosen = [r for r in rows if r.chosen]
    if chosen:
        knee = chosen[0]
        ax.scatter(
            [knee.wrong_bin_rate], [knee.rest_rate], marker="*", s=180,
            color=colors[2], edgecolors="black", linewidths=0.6, zorder=5, label="chosen",
        )
        ax.annotate(
            f"tau={knee.tau_frame:g}, votes={knee.min_votes}, high={knee.high_water:g}",
            xy=(knee.wrong_bin_rate, knee.rest_rate), xytext=(8, 8),
            textcoords="offset points", fontsize=8,
        )
    ax.set_xlabel("Wrong-bin rate")
    ax.set_ylabel("Rest-bin rate")
    ax.set_title("Consensus-threshold sweep tradeoff")
    ax.legend(loc="upper right", frameon=False)
    finalize(fig, save_path)
