"""Publication figure standards (task 014).

``setup_style`` applies the standards once: seaborn colorblind palette,
300 DPI on save, 10 pt labels / 8 pt ticks. ``finalize`` is the single
exit path every plot function uses: save PNG when asked, then
``plt.show()`` (warning-suppressed so the Agg backend stays silent) and
``plt.close()``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

DPI = 300
LABEL_SIZE = 10
TICK_SIZE = 8
PALETTE = "colorblind"

_styled = False


def setup_style() -> None:
    """Apply the publication style once (idempotent; safe under Agg)."""
    global _styled
    if _styled:
        return
    sns.set_theme(
        style="whitegrid",
        palette=PALETTE,
        rc={
            "savefig.dpi": DPI,
            "font.size": LABEL_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "axes.titlesize": LABEL_SIZE,
            "legend.fontsize": TICK_SIZE,
            "legend.title_fontsize": TICK_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
        },
    )
    _styled = True


def series_colors(names: Sequence[str]) -> dict[str, tuple[float, float, float]]:
    """Stable colorblind-palette color per series name (class, source, ...)."""
    colors = sns.color_palette(PALETTE, n_colors=max(len(names), 1))
    return {name: colors[i] for i, name in enumerate(names)}


def finalize(fig: Figure, save_path: Path | None) -> None:
    """Save as 300 DPI PNG if requested; always show then close the figure."""
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    with warnings.catch_warnings():
        # Agg has no window; show() would emit a non-interactive UserWarning.
        warnings.simplefilter("ignore", UserWarning)
        plt.show()
    plt.close(fig)
