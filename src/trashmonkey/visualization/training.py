"""Training curves from an Ultralytics run's ``results.csv``.

Left panel: box/cls/dfl losses, train solid and val dashed, one color per
loss kind. Right panel: detection metrics over epochs with the best
mAP@50-95 epoch marked (the dominant term of the Ultralytics fitness
score that ranks checkpoints). Render-only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from trashmonkey.visualization.loaders import PlotError, read_results_csv
from trashmonkey.visualization.style import finalize, series_colors, setup_style

_LOSS_KINDS = ("box_loss", "cls_loss", "dfl_loss")
_METRIC_COLUMNS = (
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "metrics/precision(B)",
    "metrics/recall(B)",
)
_BEST_METRIC = "metrics/mAP50-95(B)"


def plot_training_curves(results_csv_path: Path, save_path: Path | None = None) -> None:
    """Loss and metric curves over epochs from Ultralytics ``results.csv``."""
    setup_style()
    columns = read_results_csv(results_csv_path)
    epochs = columns["epoch"]
    losses = {
        name: values
        for name, values in columns.items()
        if name.endswith(_LOSS_KINDS) and "/" in name
    }
    metrics = {name: columns[name] for name in _METRIC_COLUMNS if name in columns}
    if not losses or not metrics:
        raise PlotError(
            f"{results_csv_path}: expected Ultralytics loss and metric columns, "
            f"got {sorted(columns)}"
        )

    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(9.0, 3.4))
    kind_colors = series_colors(_LOSS_KINDS)
    for name in sorted(losses):
        prefix, _, kind = name.partition("/")
        ax_loss.plot(
            epochs, losses[name], color=kind_colors[kind],
            ls="-" if prefix == "train" else "--", lw=1.2, label=name,
        )
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Losses (train solid, val dashed)")
    ax_loss.legend(loc="upper right", frameon=False)

    metric_colors = series_colors(sorted(metrics))
    for name in sorted(metrics):
        label = name.removeprefix("metrics/").removesuffix("(B)")
        ax_metric.plot(epochs, metrics[name], color=metric_colors[name], lw=1.4, label=label)
    if _BEST_METRIC in metrics:
        best = int(np.argmax(metrics[_BEST_METRIC]))
        ax_metric.axvline(epochs[best], color="grey", ls=":", lw=1.0)
        ax_metric.annotate(
            f"best mAP50-95\n(epoch {epochs[best]:g})", xy=(epochs[best], 0.04),
            xycoords=("data", "axes fraction"), xytext=(4, 0),
            textcoords="offset points", fontsize=8, color="grey",
        )
    ax_metric.set_xlabel("Epoch")
    ax_metric.set_ylabel("Metric")
    ax_metric.set_ylim(0.0, 1.05)
    ax_metric.set_title("Validation metrics")
    ax_metric.legend(loc="lower right", frameon=False)
    finalize(fig, save_path)
