"""Resumable, seeded data pipeline (T-final integration):

download -> remap -> autobox -> qa -> dedup -> balance -> split

Each stage wraps an existing module behind run(ctx)/is_complete(ctx); stage
manifests under data/interim/pipeline/ make reruns skip completed stages
unless --force. Entry point: ``python -m yolo_waste_sorter.data.pipeline run``
(wired to ``make repro``). Public surface re-exported here.
"""

from yolo_waste_sorter.data.pipeline.cli import build_context, main
from yolo_waste_sorter.data.pipeline.context import (
    PipelineContext,
    PipelineHalt,
    StageError,
)
from yolo_waste_sorter.data.pipeline.runner import Stage, run_pipeline
from yolo_waste_sorter.data.pipeline.stages import build_stages

__all__ = [
    "PipelineContext",
    "PipelineHalt",
    "Stage",
    "StageError",
    "build_context",
    "build_stages",
    "main",
    "run_pipeline",
]
