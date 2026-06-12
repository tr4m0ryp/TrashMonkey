"""Autobox stage: box cls-source images per class, plus the wilderness pool.

Det sources already carry labels from remap, so only images whose
`<source>__` filename prefix belongs to an annotation_type 'cls' source are
boxed. Per class the chain runs once per source (the chain's provenance is
per-source) over a symlink staging dir; generated labels land in
data/interim/autobox/<group>/ (the QA stage's pure-autobox label dirs) AND
are copied alongside the images so scan_remapped picks them up downstream.

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
from pathlib import Path

from trashmonkey.data.autobox import (
    MIN_BOX_CONFIDENCE,
    PROVENANCE_FILENAME,
    BoxRecord,
    DinoPredictFn,
    box_directory,
    build_dino_backend,
)
from trashmonkey.data.pipeline.context import PipelineContext
from trashmonkey.data.pipeline.runner import Stage
from trashmonkey.data.remap import IMAGE_SUFFIXES

WILDERNESS_GROUP = "wilderness"
WILDERNESS_PROMPT = "waste item . trash . garbage . discarded object"
STAGING_DIRNAME = ".staging"


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


def _box_class(
    ctx: PipelineContext, class_name: str, class_id: int, cls_sources: frozenset[str]
) -> list[BoxRecord]:
    class_dir = ctx.remapped_root / class_name
    by_source: dict[str, list[Path]] = {}
    for image in _images(class_dir):
        source = image.name.split("__", 1)[0]
        if source in cls_sources:
            by_source.setdefault(source, []).append(image)
    if not by_source:
        return []

    out_dir = _fresh_dir(ctx.autobox_root / class_name)
    staging_root = ctx.autobox_root / STAGING_DIRNAME / class_name
    records: list[BoxRecord] = []
    for source in sorted(by_source):
        staging = _fresh_dir(staging_root / source)
        for image in by_source[source]:
            (staging / image.name).symlink_to(image.resolve())
        records.extend(
            box_directory(
                staging,
                class_id,
                out_dir,
                class_name=class_name,
                source=source,
                dino_predict=ctx.dino_predict,
                birefnet_mask=ctx.birefnet_mask,
            )
        )
    shutil.rmtree(staging_root)
    for images in by_source.values():
        for image in images:  # alongside copies feed scan_remapped -> dedup/split
            shutil.copy2(out_dir / f"{image.stem}.txt", image.with_suffix(".txt"))
    _write_provenance(out_dir / PROVENANCE_FILENAME, records, class_dir)
    return records


def _box_wilderness(ctx: PipelineContext) -> list[BoxRecord]:
    images = _images(ctx.wilderness_root)
    if not images:
        return []
    out_dir = _fresh_dir(ctx.autobox_root / WILDERNESS_GROUP)
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
    )
    for image in images:
        shutil.copy2(out_dir / f"{image.stem}.txt", image.with_suffix(".txt"))
    _write_provenance(out_dir / PROVENANCE_FILENAME, records, ctx.wilderness_root)
    return records


def _autobox_run(ctx: PipelineContext) -> str:
    cls_sources = frozenset(
        name for name, spec in ctx.registry.items() if spec.annotation_type == "cls"
    )
    ctx.autobox_root.mkdir(parents=True, exist_ok=True)
    groups: dict[str, int] = {}
    methods: dict[str, int] = {}
    for class_id, class_name in enumerate(ctx.cfg.classes):
        records = _box_class(ctx, class_name, class_id, cls_sources)
        if records:
            groups[class_name] = len(records)
        for record in records:
            methods[record.method] = methods.get(record.method, 0) + 1
    wild = _box_wilderness(ctx)
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
