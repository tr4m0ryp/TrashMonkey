"""Per-frame detections dump the threshold tuner (012) replays (T9).

Predict runs over the VAL split only -- clean frames (severity 0) and the
degraded copies at every TEST-2 severity, because T9 tunes the rest-bin rule
on DEGRADED frames. One JSONL line per detection:
``{image_id, object_id, class_id, score, severity}`` where ``object_id`` is
the split manifest's instance-group id (all frames of one physical item share
it, so the tuner can replay per-object voting).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import yaml

from trashmonkey.models.evaluation.report import EvalError


@dataclass(frozen=True)
class ManifestIndex:
    """Split-manifest lookup: flattened image stem -> (image key, group id)."""

    key_by_stem: dict[str, str]
    group_by_key: dict[str, str]
    split_by_key: dict[str, str]


def load_manifest_index(manifest_path: Path) -> ManifestIndex:
    """Index the split manifest written by ``SplitResult.write_manifest``.

    ``emit_dataset`` flattens image keys as ``key.replace('/', '__')``, so the
    file stem of any emitted (or degraded) image maps back to its key; a stem
    collision would make detections unattributable and fails fast.
    """
    if not manifest_path.is_file():
        raise EvalError(f"split manifest not found: {manifest_path}")
    with open(manifest_path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict) or "assignments" not in raw or "groups" not in raw:
        raise EvalError(
            f"{manifest_path}: expected a split manifest with 'assignments' and 'groups'"
        )
    assignments = {str(k): str(v) for k, v in raw["assignments"].items()}
    groups = {str(k): str(v) for k, v in raw["groups"].items()}
    key_by_stem: dict[str, str] = {}
    for key in assignments:
        stem = Path(key.replace("/", "__")).stem
        if stem in key_by_stem:
            raise EvalError(
                f"image stem collision in split manifest: {key!r} vs {key_by_stem[stem]!r}"
            )
        key_by_stem[stem] = key
    return ManifestIndex(
        key_by_stem=key_by_stem, group_by_key=groups, split_by_key=assignments
    )


def image_identity(index: ManifestIndex, image_path: Path) -> tuple[str, str]:
    """Resolve an emitted image file to its (image key, instance-group id)."""
    key = index.key_by_stem.get(image_path.stem)
    if key is None:
        raise EvalError(f"image {image_path.name} has no split-manifest entry")
    group = index.group_by_key.get(key)
    if group is None:
        raise EvalError(f"image key {key!r} has no instance-group id in the manifest")
    return key, group


def dump_detections(
    model: Any,
    images: list[Path],
    severity: int,
    index: ManifestIndex,
    out: IO[str],
    *,
    conf: float,
    imgsz: int = 640,
    batch: int = 16,
) -> int:
    """Predict over ``images`` and append one JSONL line per detection.

    ``imgsz``/``batch`` are pinned so the predictor does not inherit a large
    batch from a preceding val() pass and OOM on a full-list forward.
    """
    written = 0
    results = model.predict(
        source=[str(p) for p in images],
        conf=conf,
        stream=True,
        verbose=False,
        imgsz=imgsz,
        batch=batch,
    )
    for result in results:
        image_id, object_id = image_identity(index, Path(str(result.path)))
        boxes = result.boxes
        if boxes is None:
            continue
        for class_id, score in zip(boxes.cls, boxes.conf):
            line = {
                "image_id": image_id,
                "object_id": object_id,
                "class_id": int(class_id),
                "score": float(score),
                "severity": severity,
            }
            out.write(json.dumps(line, sort_keys=True) + "\n")
            written += 1
    return written
