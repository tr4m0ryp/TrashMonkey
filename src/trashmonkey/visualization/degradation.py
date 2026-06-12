"""Degradation showcase: one clean frame beside severities 1..5.

Renders the shared deterministic degradation pipeline (``utils.degrade``)
on a single image so readers can SEE what the TEST-2 robustness tier and
the train-time corruption stack simulate. Same seed -> byte-identical
panels, so the figure is reproducible like every other artifact.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from trashmonkey.utils.degrade import degrade_image
from trashmonkey.utils.degrade.severity import MAX_SEVERITY, MIN_SEVERITY
from trashmonkey.visualization.loaders import PlotError
from trashmonkey.visualization.style import finalize, setup_style


def plot_degradation_grid(
    image_path: Path,
    save_path: Path | None = None,
    *,
    seed: int = 42,
) -> None:
    """Clean image plus the five degradation severities, one row of panels."""
    setup_style()
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise PlotError(f"could not read image: {image_path}")
    rgb = np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
    panels = [("clean", rgb)]
    panels += [
        (f"severity {s}", degrade_image(rgb, s, seed))
        for s in range(MIN_SEVERITY, MAX_SEVERITY + 1)
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(1.85 * len(panels), 2.4))
    for ax, (title, img) in zip(axes, panels, strict=True):
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(title, fontsize=8)
    finalize(fig, save_path)
