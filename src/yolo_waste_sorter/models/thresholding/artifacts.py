"""The two T9 output artifacts: thresholds.yaml and sweep.csv.

``thresholds.yaml`` (+ the exported model) is the complete deployment artifact
the runtime loads; ``sweep.csv`` is the plot-stage input with
EXACTLY the columns
``tau_frame,min_votes,high_water,wrong_bin_rate,rest_rate,chosen``.
Both writers are byte-deterministic for identical inputs.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from yolo_waste_sorter.models.thresholding.tuner import SweepCell

THRESHOLDS_FILENAME = "thresholds.yaml"
SWEEP_FILENAME = "sweep.csv"

SWEEP_COLUMNS = ("tau_frame", "min_votes", "high_water", "wrong_bin_rate", "rest_rate", "chosen")


def write_sweep_csv(cells: Sequence[SweepCell], chosen: int, path: Path) -> None:
    """One row per grid cell, chosen flagged 0/1; per-class cells report the mean tau."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(SWEEP_COLUMNS)]
    for i, cell in enumerate(cells):
        lines.append(
            f"{cell.tau_mean},{cell.min_votes},{cell.high_water},"
            f"{cell.wrong_bin_rate},{cell.rest_rate},{int(i == chosen)}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_thresholds_yaml(
    cell: SweepCell, conf_floor: float, constraint_met: bool, path: Path
) -> None:
    """Emit the deployment artifact: the selected rule + its simulated metrics."""
    data: dict[str, Any] = {
        "tau_frame": cell.tau_frame,
        "min_votes": cell.min_votes,
        "high_water": cell.high_water,
        "conf_floor": conf_floor,
        "constraint_met": constraint_met,
        "selected_metrics": {
            "wrong_bin_rate": cell.wrong_bin_rate,
            "rest_rate": cell.rest_rate,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
