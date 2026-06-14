"""Lightweight progress reporting for long pipeline steps (notebook-friendly).

The pipeline reports progress through a ``ProgressSink`` callback -- ``(label,
current, total)`` -- so the core stays UI-agnostic and the notebook decides how
to render it. ``make_progress_printer`` renders an in-place percentage line
(throttled to whole-percent changes so it never floods Colab output);
``print_stage`` prints a one-line ``[i/n] stage`` header. No third-party deps.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TextIO

# (label, current, total) -- current/total are item counts; total may be 0.
ProgressSink = Callable[[str, int, int], None]


def make_progress_printer(stream: TextIO | None = None) -> ProgressSink:
    """A ProgressSink that prints an in-place ``label: NN% (cur/total)`` line.

    Updates only when the whole-number percent changes (or on completion), so a
    22k-image stage emits at most ~100 redraws. A new label finishes the prior
    line and starts a fresh one.
    """
    out = stream if stream is not None else sys.stdout
    state = {"label": None, "pct": -1}  # type: dict[str, object]

    def sink(label: str, current: int, total: int) -> None:
        pct = int(100 * current / total) if total else 100
        if label != state["label"]:
            if state["label"] is not None:
                out.write("\n")
            state["label"] = label
            state["pct"] = -1
        done = bool(total) and current >= total
        if pct != state["pct"] or done:
            state["pct"] = pct
            out.write(f"\r  {label}: {pct:3d}%  ({current}/{total})" + ("\n" if done else ""))
            out.flush()

    return sink


def print_stage(name: str, index: int, total: int, stream: TextIO | None = None) -> None:
    """Print a ``[index/total] name`` stage header (the run_pipeline on_stage sink)."""
    out = stream if stream is not None else sys.stdout
    out.write(f"[{index}/{total}] {name}\n")
    out.flush()
