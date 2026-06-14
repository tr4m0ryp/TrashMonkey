"""Stage protocol and the resumable run loop.

Stages run in declared order; a stage whose is_complete(ctx) is satisfied is
skipped; --force re-runs from the named --stage (or all). The loop stops on
the first failure, naming the stage and a remedy hint.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from trashmonkey.data.pipeline.context import PipelineContext, PipelineHalt, StageError

logger = logging.getLogger(__name__)

RunFn = Callable[[PipelineContext], str]
CompleteFn = Callable[[PipelineContext], bool]


@dataclass(frozen=True)
class Stage:
    """One resumable pipeline step wrapping an existing module.

    `run` does the work and returns a one-line summary; `is_complete` checks
    manifests / outputs on disk; `hint` is the remedy printed on failure.
    """

    name: str
    run: RunFn
    is_complete: CompleteFn
    hint: str


def run_pipeline(
    stages: Sequence[Stage],
    ctx: PipelineContext,
    *,
    start: str | None = None,
    force: bool = False,
    on_stage: Callable[[str, int, int], None] | None = None,
) -> dict[str, str]:
    """Run the stages in order; returns {stage name: 'ran' | 'skipped'}.

    `start` begins execution at the named stage (every earlier stage must
    already be complete); `force` re-runs every executed stage even when its
    is_complete(ctx) is satisfied. `on_stage(name, index, total)` fires as each
    stage (1-based) is reached -- a coarse progress signal for the notebook.
    PipelineHalt (the QA gate) propagates untouched so the CLI can print the
    review-queue path.
    """
    names = [stage.name for stage in stages]
    if len(set(names)) != len(names):
        raise StageError(f"duplicate stage names: {names}")
    if start is not None and start not in names:
        raise StageError(f"unknown stage '{start}'; stages in order: {', '.join(names)}")
    begin = 0 if start is None else names.index(start)
    total = len(stages)

    def announce(stage: Stage, index: int) -> None:
        if on_stage is not None:
            on_stage(stage.name, index, total)

    actions: dict[str, str] = {}
    for index, stage in enumerate(stages[:begin], start=1):
        if not stage.is_complete(ctx):
            raise StageError(
                f"stage '{stage.name}' is not complete but precedes --stage {start}; "
                f"run the pipeline without --stage first"
            )
        announce(stage, index)
        actions[stage.name] = "skipped"
        logger.info("stage %s: skipped (complete, precedes --stage %s)", stage.name, start)
    for offset, stage in enumerate(stages[begin:]):
        announce(stage, begin + offset + 1)
        if not force and stage.is_complete(ctx):
            actions[stage.name] = "skipped"
            logger.info("stage %s: skipped (already complete)", stage.name)
            continue
        try:
            summary = stage.run(ctx)
        except PipelineHalt:
            raise
        except Exception as exc:
            raise StageError(
                f"stage '{stage.name}' failed: {exc} -- hint: {stage.hint}"
            ) from exc
        actions[stage.name] = "ran"
        logger.info("stage %s: %s", stage.name, summary)
    return actions
