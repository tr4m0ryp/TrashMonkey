"""Label-quality filter (T4): drop low-quality auto-labels before balancing.

Public surface re-exported here; ``filter`` submodule is implementation detail.
"""

from .filter import (
    ALL_REASONS,
    REASON_BOX_TOO_LARGE,
    REASON_BOX_TOO_SMALL,
    REASON_LOW_CONFIDENCE,
    REASON_METHOD,
    FilterResult,
    filter_items,
)

__all__ = [
    "ALL_REASONS",
    "REASON_BOX_TOO_LARGE",
    "REASON_BOX_TOO_SMALL",
    "REASON_LOW_CONFIDENCE",
    "REASON_METHOD",
    "FilterResult",
    "filter_items",
]
