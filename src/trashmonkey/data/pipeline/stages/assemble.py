"""Dedup, balance, and split stage wrappers over the interim layout.

Each stage re-scans data/interim/remapped/ and narrows it through the previous
stage's manifest (dedup kept keys -> balance kept keys), so a resumed run
reconstructs exactly the same item pool without in-memory hand-off. Source
priority for dedup is the configs/datasets.yaml file order (T10); the TEST-1
leave-out source bypasses balancing and lands wholesale in the test split.
"""

from __future__ import annotations

from typing import Any

from trashmonkey.data.balance import balance_items
from trashmonkey.data.dedup import Item, NearEdge, dedup_items, scan_remapped
from trashmonkey.data.pipeline.context import (
    PipelineContext,
    StageError,
    manifest_str_list,
)
from trashmonkey.data.pipeline.runner import Stage
from trashmonkey.data.split import emit_dataset, split_items


def _scan(ctx: PipelineContext) -> list[Item]:
    return scan_remapped(ctx.remapped_root, list(ctx.cfg.classes))


def _items_from_kept(ctx: PipelineContext, stage: str) -> list[Item]:
    """Re-scan the interim layout narrowed to a previous stage's kept keys."""
    kept = set(manifest_str_list(ctx.read_manifest(stage), "kept", stage))
    items = [item for item in _scan(ctx) if item.key in kept]
    missing = kept - {item.key for item in items}
    if missing:
        raise StageError(
            f"{len(missing)} image(s) from the '{stage}' manifest are gone from "
            f"{ctx.remapped_root}, e.g. {sorted(missing)[:3]} -- rerun with "
            f"--stage {stage} --force after fixing the interim tree"
        )
    return items


def _dedup_run(ctx: PipelineContext) -> str:
    result = dedup_items(_scan(ctx), list(ctx.registry))
    result.write_manifest(ctx.manifest_path("dedup"))
    return (
        f"kept {len(result.kept)}, dropped {len(result.dropped)} exact duplicates, "
        f"{len(result.near_edges)} near-duplicate edges"
    )


def dedup_stage() -> Stage:
    return Stage(
        name="dedup",
        run=_dedup_run,
        is_complete=lambda ctx: ctx.manifest_path("dedup").is_file(),
        hint=(
            "ensure remap and autobox completed and every file under "
            "data/interim/remapped/<class>/ follows the '<source>__<name>' convention"
        ),
    )


def _balance_run(ctx: PipelineContext) -> str:
    items = _items_from_kept(ctx, "dedup")
    leave_out = ctx.cfg.eval.leave_out_source
    result = balance_items(
        items,
        seed=ctx.cfg.seed,
        exempt_sources=frozenset({leave_out}) if leave_out else frozenset(),
        source_caps={name: spec.cap for name, spec in ctx.registry.items() if spec.cap},
    )
    result.write_manifest(ctx.manifest_path("balance"))
    summary = f"kept {len(result.kept)} of {len(items)} images (cap {result.cap})"
    if result.floor_warnings:
        summary += f", {len(result.floor_warnings)} class(es) under the {result.floor} floor"
    return summary


def balance_stage() -> Stage:
    return Stage(
        name="balance",
        run=_balance_run,
        is_complete=lambda ctx: ctx.manifest_path("balance").is_file(),
        hint="if the dedup manifest is stale, rerun with --stage dedup --force",
    )


def _near_edges(manifest: dict[str, Any]) -> list[NearEdge]:
    raw = manifest.get("near_edges")
    if not isinstance(raw, list):
        raise StageError("'dedup' manifest: 'near_edges' must be a list")
    edges: list[NearEdge] = []
    for entry in raw:
        if not isinstance(entry, list) or len(entry) != 3:
            raise StageError(f"'dedup' manifest: malformed near edge {entry!r}")
        edges.append(NearEdge(key_a=str(entry[0]), key_b=str(entry[1]), distance=int(entry[2])))
    return edges


def _split_run(ctx: PipelineContext) -> str:
    items = _items_from_kept(ctx, "balance")
    edges = _near_edges(ctx.read_manifest("dedup"))
    result = split_items(
        items,
        edges,
        leave_out_source=ctx.cfg.eval.leave_out_source,
        val_fraction=ctx.cfg.eval.val_fraction,
        seed=ctx.cfg.seed,
    )
    result.write_manifest(ctx.manifest_path("split"))
    yaml_path = emit_dataset(
        result, items, ctx.processed_root, ctx.cfg.experiment.name, list(ctx.cfg.classes)
    )
    counts = ", ".join(f"{split}={sum(c.values())}" for split, c in result.counts.items())
    return f"dataset at {yaml_path} ({counts})"


def _split_complete(ctx: PipelineContext) -> bool:
    data_yaml = ctx.processed_root / ctx.cfg.experiment.name / "dataset.yaml"
    return ctx.manifest_path("split").is_file() and data_yaml.is_file()


def split_stage() -> Stage:
    return Stage(
        name="split",
        run=_split_run,
        is_complete=_split_complete,
        hint=(
            "check eval.leave_out_source / eval.val_fraction in configs/config.yaml; "
            "a missing-label error means autobox did not cover a cls source"
        ),
    )
