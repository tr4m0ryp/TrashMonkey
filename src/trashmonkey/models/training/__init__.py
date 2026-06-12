"""Training entrypoint (task 009): T7 recipe + T5 stack + runs.jsonl logging.

Public surface re-exported here; ``trashmonkey.models.train`` is the
thin CLI module on top of this package.
"""

from trashmonkey.models.training.core import (
    DEFAULT_RUNS_JSONL,
    RunResult,
    smoke_requested,
    train,
)
from trashmonkey.models.training.escalation import check_escalation, extract_metrics
from trashmonkey.models.training.guards import (
    MIN_ULTRALYTICS,
    require_augmentations_support,
    validate_train_config,
)
from trashmonkey.models.training.kwargs import SMOKE_OVERRIDES, build_train_kwargs
from trashmonkey.models.training.resume import checkpoint_epoch, find_resumable
from trashmonkey.models.training.runlog import append_run_record, build_run_record
from trashmonkey.models.training.smoke import build_smoke_dataset

__all__ = [
    "DEFAULT_RUNS_JSONL",
    "MIN_ULTRALYTICS",
    "SMOKE_OVERRIDES",
    "RunResult",
    "append_run_record",
    "build_run_record",
    "build_smoke_dataset",
    "build_train_kwargs",
    "check_escalation",
    "checkpoint_epoch",
    "extract_metrics",
    "find_resumable",
    "require_augmentations_support",
    "smoke_requested",
    "train",
    "validate_train_config",
]
