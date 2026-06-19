"""Autobox stage: box cls-source images per class, plus the wilderness pool.

Det sources already carry labels from remap, so only images whose
`<source>__` filename prefix belongs to an annotation_type 'cls' source are
boxed. Per class the chain runs once per source (the chain's provenance is
per-source) over a symlink staging dir; generated labels land in
data/interim/autobox/<group>/ (the QA stage's pure-autobox label dirs) AND
are copied alongside the images so scan_remapped picks them up downstream.

Resume: each (class, source) group -- and the wilderness pool -- writes its
BoxRecords to a checkpoint under data/interim/autobox/.checkpoints/ as soon as
it finishes. A re-run after a crash (the stage manifest is absent) loads those
checkpoints and skips the boxing, so a disconnect costs at most the in-flight
group, not the whole stage. A forced re-run (the manifest IS present, i.e. the
runner bypassed is_complete) wipes the checkpoints and re-boxes from scratch.

The wilderness pool (data/interim/wilderness/) is boxed for localization
only: placeholder class 0, provenance source 'wilderness', a generic prompt.

Interface bridge (deliberate): the chain's raw provenance JSONL stores bare
image names and null confidences, which qa.load_provenance cannot consume
(float(None) raises; emit_review_queue needs resolvable paths). The merged
per-group provenance written here uses absolute image paths and 0.0 for
missing confidences, so birefnet/centerbox records sort first in review.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from trashmonkey.data.autobox import (
    MIN_BOX_CONFIDENCE,
    PROMPTS,
    PROVENANCE_FILENAME,
    BoxRecord,
    Detection,
    DinoPredictFn,
    MaskFn,
    Method,
    ProgressFn,
    box_directory,
    build_birefnet_backend,
    build_dino_backend,
)
from trashmonkey.data.pipeline.context import PipelineContext
from trashmonkey.data.pipeline.runner import Stage
from trashmonkey.data.remap import IMAGE_SUFFIXES

WILDERNESS_GROUP = "wilderness"
WILDERNESS_PROMPT = "waste item . trash . garbage . discarded object"
STAGING_DIRNAME = ".staging"
CHECKPOINTS_DIRNAME = ".checkpoints"


def _images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def _fresh_dir(directory: Path) -> Path:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    return directory


def _write_provenance(path: Path, records: list[BoxRecord], image_dir: Path) -> None:
    """Merged provenance with absolute image paths and float confidences."""
    lines = []
    for record in records:
        payload = record.to_dict()
        payload["image"] = str((image_dir / record.image).resolve())
        payload["confidence"] = record.confidence if record.confidence is not None else 0.0
        lines.append(json.dumps(payload, sort_keys=True))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- per-group resume checkpoints ------------------------------------------------


def _checkpoint_path(ctx: PipelineContext, key: str) -> Path:
    return ctx.autobox_root / CHECKPOINTS_DIRNAME / f"{key}.jsonl"


def _save_records(path: Path, records: list[BoxRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(record.to_json() for record in records)
    path.write_text(body + ("\n" if body else ""), encoding="utf-8")


def _load_records(path: Path) -> list[BoxRecord]:
    records: list[BoxRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        records.append(
            BoxRecord(
                image=data["image"],
                source=data["source"],
                method=cast(Method, data["method"]),
                confidence=data["confidence"],
                flags=list(data.get("flags", [])),
            )
        )
    return records


def _reset_autobox(ctx: PipelineContext) -> None:
    """Forced re-run: drop every checkpoint and prior output for a clean re-box."""
    if ctx.autobox_root.exists():
        shutil.rmtree(ctx.autobox_root)
    ctx.autobox_root.mkdir(parents=True, exist_ok=True)


# --- boxing ----------------------------------------------------------------------


def _class_image_count(ctx: PipelineContext, class_name: str, cls_sources: frozenset[str]) -> int:
    """How many cls-source images this class contributes (for progress totals)."""
    return sum(
        1
        for image in _images(ctx.remapped_root / class_name)
        if image.name.split("__", 1)[0] in cls_sources
    )


def _box_class(
    ctx: PipelineContext,
    class_name: str,
    class_id: int,
    cls_sources: frozenset[str],
    dino: DinoPredictFn,
    birefnet: MaskFn,
    on_image: ProgressFn | None = None,
    advance: Callable[[int], None] | None = None,
) -> list[BoxRecord]:
    class_dir = ctx.remapped_root / class_name
    by_source: dict[str, list[Path]] = {}
    for image in _images(class_dir):
        source = image.name.split("__", 1)[0]
        if source in cls_sources:
            by_source.setdefault(source, []).append(image)
    if not by_source:
        return []

    out_dir = ctx.autobox_root / class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    staging_root = ctx.autobox_root / STAGING_DIRNAME / class_name
    records: list[BoxRecord] = []
    for source in sorted(by_source):
        checkpoint = _checkpoint_path(ctx, f"{class_name}__{source}")
        if checkpoint.is_file():
            group = _load_records(checkpoint)
            if advance is not None:
                advance(len(group))  # labels already written alongside images last run
        else:
            staging = _fresh_dir(staging_root / source)
            for image in by_source[source]:
                (staging / image.name).symlink_to(image.resolve())
            group = box_directory(
                staging,
                class_id,
                out_dir,
                class_name=class_name,
                source=source,
                # Per-source method order from the registry (() = pipeline default).
                box_order=cast("tuple[Method, ...]", ctx.registry[source].box_order),
                dino_predict=dino,  # shared, built once per class
                birefnet_mask=birefnet,  # shared, built once globally
                progress=on_image,
            )
            for image in by_source[source]:  # alongside copies feed scan_remapped
                shutil.copy2(out_dir / f"{image.stem}.txt", image.with_suffix(".txt"))
            _save_records(checkpoint, group)  # checkpoint LAST: its presence => labels done
        records.extend(group)
    if staging_root.exists():
        shutil.rmtree(staging_root)
    _write_provenance(out_dir / PROVENANCE_FILENAME, records, class_dir)
    return records


def _box_wilderness(
    ctx: PipelineContext,
    on_image: ProgressFn | None = None,
    advance: Callable[[int], None] | None = None,
) -> list[BoxRecord]:
    images = _images(ctx.wilderness_root)
    if not images:
        return []
    out_dir = ctx.autobox_root / WILDERNESS_GROUP
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _checkpoint_path(ctx, WILDERNESS_GROUP)
    if checkpoint.is_file():
        records = _load_records(checkpoint)
        if advance is not None:
            advance(len(records))
    else:
        dino: DinoPredictFn = ctx.dino_predict or build_dino_backend(
            WILDERNESS_PROMPT, box_threshold=MIN_BOX_CONFIDENCE
        )
        # class_name only gates the chain's PROMPTS membership check here: the
        # injected dino backend carries the wilderness prompt, and class 0 is a
        # placeholder -- the T9 threshold tuner needs localization only.
        records = box_directory(
            ctx.wilderness_root,
            0,
            out_dir,
            class_name=ctx.cfg.classes[0],
            source=WILDERNESS_GROUP,
            dino_predict=dino,
            birefnet_mask=ctx.birefnet_mask,
            progress=on_image,
        )
        for image in images:
            shutil.copy2(out_dir / f"{image.stem}.txt", image.with_suffix(".txt"))
        _save_records(checkpoint, records)
    _write_provenance(out_dir / PROVENANCE_FILENAME, records, ctx.wilderness_root)
    return records


def _autobox_run(ctx: PipelineContext) -> str:
    cls_sources = frozenset(
        name for name, spec in ctx.registry.items() if spec.annotation_type == "cls"
    )
    ctx.autobox_root.mkdir(parents=True, exist_ok=True)
    # The manifest existing while run() is called means a FORCED re-run (the
    # runner skips completed stages otherwise) -> reset for a clean re-box.
    # Absent manifest + present checkpoints => resume after a crash.
    if ctx.manifest_path("autobox").is_file():
        _reset_autobox(ctx)

    # One cumulative percentage across the whole stage (the long pole of the
    # build): grand total = every cls-source image plus the wilderness pool.
    grand_total = sum(
        _class_image_count(ctx, class_name, cls_sources) for class_name in ctx.cfg.classes
    ) + len(_images(ctx.wilderness_root))
    done = {"n": 0}

    def _report() -> None:
        if ctx.progress is not None and grand_total:
            ctx.progress("autobox", done["n"], grand_total)

    def on_image(_done: int, _total: int, _path: Path) -> None:
        done["n"] += 1
        _report()

    def advance(count: int) -> None:  # account checkpoint-skipped images in the bar
        done["n"] += count
        _report()

    groups: dict[str, int] = {}
    methods: dict[str, int] = {}
    for class_id, class_name in enumerate(ctx.cfg.classes):
        records = _box_class(ctx, class_name, class_id, cls_sources, on_image, advance)
        if records:
            groups[class_name] = len(records)
        for record in records:
            methods[record.method] = methods.get(record.method, 0) + 1
    wild = _box_wilderness(ctx, on_image, advance)
    if wild:
        groups[WILDERNESS_GROUP] = len(wild)
    for record in wild:
        methods[record.method] = methods.get(record.method, 0) + 1
    ctx.write_manifest(
        "autobox", {"groups": groups, "methods": dict(sorted(methods.items()))}
    )
    return f"{sum(groups.values())} images auto-boxed across {len(groups)} group(s)"


def autobox_stage() -> Stage:
    return Stage(
        name="autobox",
        run=_autobox_run,
        is_complete=lambda ctx: ctx.manifest_path("autobox").is_file(),
        hint=(
            "install the boxing extra (pip install 'trashmonkey[boxing]') for the "
            "real backends, and check data/interim/remapped/ exists (remap stage)"
        ),
    )
