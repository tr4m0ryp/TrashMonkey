"""The T9 multi-frame consensus rule -- the ONE pure decision function.

The Jetson runtime (task 015) imports ``consensus_decision``, ``REST`` and
``ThresholdParams`` from here; nothing in this module touches files, config,
or the simulator. "rest" is NOT a trained class (vision C3): it exists only
as the rejection outcome of this rule.

Rule (T9): a frame casts a qualified vote for class ``c`` when its top
detection is ``c`` with score >= tau_frame (per-class tau supported). The
object sorts to bin ``c`` iff over its sightings (a) qualified votes for
``c`` >= min_votes, (b) ``c`` holds a STRICT majority of all qualified
votes, and (c) the max single-frame score for ``c`` >= high_water.
Otherwise the object goes to the rest bin. Consensus converts per-frame
error to per-object error roughly binomially (F13/F14), which is why a
permissive ``conf_floor`` per frame plus this agreement rule beats one high
single-frame threshold.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final


class ThresholdError(Exception):
    """Threshold inputs (params, detections, manifest) are malformed."""


class RestType:
    """Singleton sentinel for the rest-bin decision (NOT a class id)."""

    _instance: RestType | None = None

    def __new__(cls) -> RestType:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "REST"


REST: Final[RestType] = RestType()

# A decision is a trained class id (index into config classes) or REST.
Decision = int | RestType

# One sighting's top detection: (class_id, score).
Vote = tuple[int, float]


@dataclass(frozen=True)
class ThresholdParams:
    """The deployable T9 rule parameters (thresholds.yaml contents).

    ``tau_frame`` is a single global threshold or a per-class mapping
    ``class_id -> tau`` (per-class mode triggers when the precision>=0.95
    confidence spans > 0.1 across classes, F12). ``conf_floor`` is the
    permissive per-frame detector confidence; it is carried for the runtime
    (the detector itself filters at it) and is not re-applied here.
    """

    tau_frame: float | Mapping[int, float]
    min_votes: int
    high_water: float
    conf_floor: float

    def tau_for(self, class_id: int) -> float:
        """Per-class qualified-vote threshold; fail fast on unmapped ids."""
        if isinstance(self.tau_frame, Mapping):
            try:
                return self.tau_frame[class_id]
            except KeyError:
                raise ThresholdError(
                    f"class_id {class_id} has no per-class tau_frame entry "
                    f"(mapped: {sorted(self.tau_frame)})"
                ) from None
        return self.tau_frame


def consensus_decision(votes: Sequence[Vote], params: ThresholdParams) -> Decision:
    """Apply the T9 rule to one object's sightings; pure and deterministic.

    ``votes`` holds one ``(class_id, score)`` entry per sighting -- the
    frame's TOP detection. At most one class can hold a strict majority of
    qualified votes, so the decision is unambiguous. Empty or all-unqualified
    sightings go to REST.
    """
    if params.min_votes < 1:
        raise ThresholdError(f"min_votes must be >= 1, got {params.min_votes}")
    qualified: dict[int, int] = {}
    for class_id, score in votes:
        if score >= params.tau_for(class_id):
            qualified[class_id] = qualified.get(class_id, 0) + 1
    if not qualified:
        return REST
    total = sum(qualified.values())
    # Ties for the top count can never hold a strict majority -> REST anyway.
    top = min(qualified, key=lambda c: (-qualified[c], c))
    if qualified[top] < params.min_votes:
        return REST
    if 2 * qualified[top] <= total:
        return REST  # not a STRICT majority of all qualified votes
    high = max((score for class_id, score in votes if class_id == top), default=0.0)
    if high < params.high_water:
        return REST
    return top
