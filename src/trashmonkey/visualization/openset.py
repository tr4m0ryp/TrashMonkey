"""Open-set score separation: trained-class frames vs probe-pool frames.

The figure that motivates the consensus rule: per-frame top-1 confidence
distributions for images of trained classes (the evaluation detections
dump) and for open-set probe images the model should reject (the
wilderness dump). Overlap right of the per-frame threshold is exactly the
leak a single confidence check cannot close. Render-only; the per-frame
top-1 grouping mirrors the consensus rule's qualified-vote input.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from trashmonkey.visualization.loaders import DetectionLine, read_detections_jsonl
from trashmonkey.visualization.style import finalize, setup_style


def top_scores_per_frame(lines: tuple[DetectionLine, ...]) -> tuple[float, ...]:
    """Max detection score per (image, severity) frame -- the vote input."""
    best: dict[tuple[str, int], float] = {}
    for line in lines:
        key = (line.image_id, line.severity)
        if line.score > best.get(key, -1.0):
            best[key] = line.score
    return tuple(best[key] for key in sorted(best))


def plot_confidence_separation(
    detections_path: Path,
    wilderness_path: Path,
    save_path: Path | None = None,
    *,
    tau_frame: float | None = None,
) -> None:
    """Overlaid per-frame top-1 confidence histograms, known vs open-set."""
    setup_style()
    known = top_scores_per_frame(read_detections_jsonl(detections_path))
    probes = top_scores_per_frame(read_detections_jsonl(wilderness_path))
    colors = sns.color_palette("colorblind", n_colors=3)
    bins = [float(edge) for edge in np.linspace(0.0, 1.0, 41)]

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.hist(
        known, bins=bins, density=True, alpha=0.6, color=colors[0],
        label=f"trained classes (n={len(known)})",
    )
    ax.hist(
        probes, bins=bins, density=True, alpha=0.6, color=colors[1],
        label=f"open-set probes (n={len(probes)})",
    )
    if tau_frame is not None:
        ax.axvline(tau_frame, color="grey", ls="--", lw=1.0)
        ax.annotate(
            f"tau_frame {tau_frame:g}", xy=(tau_frame, 1.0),
            xycoords=("data", "axes fraction"), xytext=(3, -10),
            textcoords="offset points", fontsize=8, color="grey",
        )
    ax.set_xlabel("Top-1 confidence per frame")
    ax.set_ylabel("Density")
    ax.set_xlim(0.0, 1.0)
    ax.set_title("Open-set score separation")
    ax.legend(loc="upper left", frameon=False)
    finalize(fig, save_path)
