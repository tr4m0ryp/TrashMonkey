"""Programmatic one-call dataset build (the ``make repro`` path, importable).

``build_dataset`` is the in-process equivalent of
``python -m trashmonkey.data.pipeline run``: it loads the config + sibling
registry, seeds, and runs every stage in order (download -> remap -> autobox ->
qa -> dedup -> balance -> split). The notebook calls this so a fresh Colab
session can fetch and assemble the dataset itself instead of expecting a
pre-built archive.

``ack_review=True`` is REQUIRED for an unattended build: the QA stage raises
``PipelineHalt`` to force human review of its queue otherwise. Acknowledging it
is the documented continue mechanism -- it skips the manual review, not the QA
checks themselves.
"""

from __future__ import annotations

from pathlib import Path

from trashmonkey.data.pipeline.cli import build_context
from trashmonkey.data.pipeline.runner import run_pipeline
from trashmonkey.data.pipeline.stages import build_stages
from trashmonkey.utils.seed import set_seed


def build_dataset(
    config_path: Path,
    *,
    ack_review: bool = True,
    start: str | None = None,
    force: bool = False,
) -> dict[str, str]:
    """Run the full data pipeline in-process; return {stage: 'ran'|'skipped'}.

    Completed stages are skipped (resumable), so re-invoking after an
    interrupted build resumes where it stopped. ``start``/``force`` mirror the
    CLI flags. Raises ``PipelineHalt`` if ``ack_review`` is False and the QA
    gate trips; ``StageError`` on any stage failure.
    """
    ctx = build_context(config_path, ack_review=ack_review)
    set_seed(ctx.cfg.seed)
    return run_pipeline(build_stages(), ctx, start=start, force=force)
