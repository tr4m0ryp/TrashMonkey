"""Instance-grouped, stratified split into train/val + held-out eval tiers (T6/T5).

TEST-1 is ALL images of ``eval.leave_out_source`` (config; 'realwaste' after the
census), excluded from train/val entirely; a null value warns loudly and skips
TEST-1. ``role: test_only`` sources land wholesale in a ``wild_test`` tier,
likewise excluded from train/val. A group-aware, (source, class)-stratified
``eval.clean_holdout`` fraction of the named clean sources is carved into a
``clean_test`` tier. The remaining pool splits train/val stratified by source x
class on top of instance groups: connected components over the dedup stage's
near-duplicate graph are "the same physical object" and NEVER straddle a split
boundary. ``emit_dataset`` writes the final YOLO detect layout
(``data/processed/<experiment>/{images,labels}/<split>/``) plus ``dataset.yaml``
with names in config class order, emitting only the splits that have members.
"""

from __future__ import annotations

from .assign import split_items
from .emit import emit_dataset
from .grouping import group_instances, primary_stratum
from .result import (
    DEFAULT_VAL_FRACTION,
    SPLITS,
    SplitError,
    SplitResult,
)

__all__ = [
    "DEFAULT_VAL_FRACTION",
    "SPLITS",
    "SplitError",
    "SplitResult",
    "emit_dataset",
    "group_instances",
    "primary_stratum",
    "split_items",
]
