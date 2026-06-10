"""Declared stage order: download -> remap -> autobox -> qa -> dedup -> balance -> split."""

from yolo_waste_sorter.data.pipeline.runner import Stage
from yolo_waste_sorter.data.pipeline.stages.assemble import (
    balance_stage,
    dedup_stage,
    split_stage,
)
from yolo_waste_sorter.data.pipeline.stages.autobox import autobox_stage
from yolo_waste_sorter.data.pipeline.stages.ingest import download_stage, remap_stage
from yolo_waste_sorter.data.pipeline.stages.qa import qa_stage

__all__ = [
    "autobox_stage",
    "balance_stage",
    "build_stages",
    "dedup_stage",
    "download_stage",
    "qa_stage",
    "remap_stage",
    "split_stage",
]


def build_stages() -> tuple[Stage, ...]:
    """The full pipeline in execution order."""
    return (
        download_stage(),
        remap_stage(),
        autobox_stage(),
        qa_stage(),
        dedup_stage(),
        balance_stage(),
        split_stage(),
    )
