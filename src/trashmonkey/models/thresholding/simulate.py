"""Consensus simulation inputs: detections JSONL, truth mapping, sightings.

Replays the evaluation stage's per-frame detections dump (one line per
detection: ``{image_id, object_id, class_id, score, severity}``; severity 0
is the clean copy) as M emulated sightings per object, M ~ Uniform(5, 15)
per T9's "5-15 sightings" conveyor window. Tuning happens on DEGRADED frames
(severity >= 1, T9); clean frames are used only for objects that have no
degraded sightings at all (loudly logged).
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from trashmonkey.models.evaluation.detections import load_manifest_index
from trashmonkey.models.thresholding.consensus import ThresholdError, Vote

logger = logging.getLogger(__name__)

SIGHTINGS_RANGE = (5, 15)  # T9: per-object sightings in one conveyor window

_DETECTION_FIELDS = ("image_id", "object_id", "class_id", "score", "severity")


@dataclass(frozen=True)
class Detection:
    """One JSONL line of the evaluation detections dump."""

    image_id: str
    object_id: str
    class_id: int
    score: float
    severity: int


def load_detections(path: Path) -> list[Detection]:
    """Parse a detections JSONL; fail fast on missing or mistyped fields."""
    if not path.is_file():
        raise ThresholdError(f"detections file not found: {path}")
    detections: list[Detection] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            where = f"{path}:{lineno}"
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ThresholdError(f"{where}: invalid JSON: {exc}") from exc
            if not isinstance(raw, dict):
                raise ThresholdError(f"{where}: expected an object, got {type(raw).__name__}")
            missing = [k for k in _DETECTION_FIELDS if k not in raw]
            if missing:
                raise ThresholdError(f"{where}: missing field(s): {', '.join(missing)}")
            try:
                detections.append(
                    Detection(
                        image_id=str(raw["image_id"]),
                        object_id=str(raw["object_id"]),
                        class_id=int(raw["class_id"]),
                        score=float(raw["score"]),
                        severity=int(raw["severity"]),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ThresholdError(f"{where}: mistyped field: {exc}") from exc
    return detections


def truth_from_manifest(manifest_path: Path, classes: Sequence[str]) -> dict[str, int]:
    """Derive ``object_id -> true class_id`` for VAL objects from the split manifest.

    Image keys follow the remap convention ``<class>/<source>__<name>``
    (data/dedup.py), so the ground-truth class of every val image is its
    key's first path component; the object_id is the manifest's instance
    -group id. A group mixing classes is a pipeline defect -> fail fast.
    """
    index = load_manifest_index(manifest_path)
    class_ids = {name: class_id for class_id, name in enumerate(classes)}
    truth: dict[str, int] = {}
    for key, split in sorted(index.split_by_key.items()):
        if split != "val":
            continue
        class_name, _, rest = key.partition("/")
        if not rest or class_name not in class_ids:
            raise ThresholdError(
                f"image key {key!r} does not start with a known class "
                f"(expected '<class>/<source>__<name>' with class in {list(classes)})"
            )
        object_id = index.group_by_key.get(key)
        if object_id is None:
            raise ThresholdError(f"image key {key!r} has no instance-group id in the manifest")
        class_id = class_ids[class_name]
        known = truth.setdefault(object_id, class_id)
        if known != class_id:
            raise ThresholdError(
                f"instance group {object_id!r} mixes classes "
                f"{classes[known]!r} and {class_name!r} -- split stage defect"
            )
    if not truth:
        raise ThresholdError(f"{manifest_path}: no val images -- nothing to tune on")
    return truth


def top_votes_per_frame(detections: Sequence[Detection]) -> dict[str, list[tuple[int, Vote]]]:
    """Per object: each frame's top detection as ``(severity, (class_id, score))``.

    A frame is one (image_id, severity) pair -- every degraded copy of a val
    image is a distinct sighting. The top detection is the frame's highest
    score (ties break on the smaller class_id for determinism).
    """
    frames: dict[str, dict[tuple[str, int], Detection]] = {}
    for det in detections:
        per_object = frames.setdefault(det.object_id, {})
        frame_key = (det.image_id, det.severity)
        best = per_object.get(frame_key)
        if best is None or (-det.score, det.class_id) < (-best.score, best.class_id):
            per_object[frame_key] = det
    return {
        object_id: [
            (det.severity, (det.class_id, det.score))
            for _, det in sorted(per_frame.items())
        ]
        for object_id, per_frame in frames.items()
    }


def sample_sightings(
    object_id: str,
    frames: Sequence[tuple[int, Vote]],
    seed: int,
    *,
    sightings_range: tuple[int, int] = SIGHTINGS_RANGE,
) -> list[Vote]:
    """Emulate one conveyor pass: M ~ Uniform(5, 15) sightings with replacement.

    The pool is the object's DEGRADED frames (severity >= 1); clean frames
    only back-fill objects with zero degraded sightings. Seeding hashes the
    object_id into the stream so the draw is independent of iteration order.
    An empty pool yields no votes (-> REST downstream).
    """
    pool = [vote for severity, vote in frames if severity >= 1]
    if not pool:
        pool = [vote for _, vote in frames]
    if not pool:
        return []
    rng = random.Random(f"{seed}:thresholds:{object_id}")
    count = rng.randint(*sightings_range)
    return [pool[rng.randrange(len(pool))] for _ in range(count)]


def build_sightings(
    detections: Sequence[Detection],
    object_ids: Sequence[str],
    seed: int,
) -> dict[str, list[Vote]]:
    """Sampled sightings for every object in ``object_ids`` (no-detection -> [])."""
    frames = top_votes_per_frame(detections)
    clean_only = [
        object_id
        for object_id in object_ids
        if frames.get(object_id) and not any(s >= 1 for s, _ in frames[object_id])
    ]
    if clean_only:
        logger.warning(
            "thresholds: %d object(s) have no degraded sightings; sampling their "
            "clean frames instead (T9 prefers severity >= 1), e.g. %s",
            len(clean_only),
            clean_only[:5],
        )
    return {
        object_id: sample_sightings(object_id, frames.get(object_id, []), seed)
        for object_id in sorted(object_ids)
    }


def check_universe(
    detections: Sequence[Detection],
    truth: Mapping[str, int],
    *,
    where: str,
) -> None:
    """Every detection's object_id must be a known val object (011 contract)."""
    unknown = sorted({d.object_id for d in detections} - set(truth))
    if unknown:
        raise ThresholdError(
            f"{where}: {len(unknown)} object_id(s) missing from the truth mapping, "
            f"e.g. {unknown[:5]} -- the detections dump must cover val objects only"
        )
