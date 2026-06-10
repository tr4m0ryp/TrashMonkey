"""Dataset composition: per-class image counts stacked by source, per split.

Reads the split-stage manifest (``data/split.py``); the per-source axis is
recovered from the ``<class>/<source>__<name>`` assignment keys.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from yolo_waste_sorter.data.split import SPLITS
from yolo_waste_sorter.visualization.loaders import read_split_composition
from yolo_waste_sorter.visualization.style import finalize, series_colors, setup_style


def plot_dataset_composition(
    split_manifest_path: Path,
    save_path: Path | None = None,
) -> None:
    """Stacked bars (class x source), one facet per split (train/val/test)."""
    setup_style()
    counts = read_split_composition(split_manifest_path)
    splits = [s for s in SPLITS if s in counts]
    splits += sorted(set(counts) - set(splits))  # tolerate extra split names
    classes = sorted({name for per_class in counts.values() for name in per_class})
    sources = sorted(
        {src for per_class in counts.values() for srcs in per_class.values() for src in srcs}
    )
    colors = series_colors(sources)

    fig, axes = plt.subplots(
        1, len(splits), figsize=(1.1 + 2.4 * len(splits), 3.4), sharey=True, squeeze=False
    )
    positions = np.arange(len(classes))
    for ax, split in zip(axes[0], splits, strict=True):
        bottom = np.zeros(len(classes))
        for source in sources:
            heights = np.array(
                [float(counts[split].get(name, {}).get(source, 0)) for name in classes]
            )
            ax.bar(positions, heights, bottom=bottom, width=0.7, color=colors[source])
            bottom += heights
        total = int(bottom.sum())
        ax.set_title(f"{split} (n={total})")
        ax.set_xticks(positions)
        ax.set_xticklabels(classes, rotation=45, ha="right")
    axes[0][0].set_ylabel("Images")
    handles = [Patch(facecolor=colors[source]) for source in sources]
    fig.legend(
        handles, sources, title="Source", loc="upper right",
        bbox_to_anchor=(0.99, 0.98), frameon=False,
    )
    finalize(fig, save_path)
