"""Download and remap stage wrappers.

Both modules keep their own per-source manifests (data/raw/<source>/
.manifest.json and data/interim/remapped/.manifests/<source>.yaml), so stage
completion is "every registered source has its manifest". A forced re-run of
the download stage re-verifies checksums but never refetches a verified
archive: data/raw/ is append-only (use the download CLI --force to refetch).
"""

from __future__ import annotations

from yolo_waste_sorter.data.download import MANIFEST_NAME, download_sources
from yolo_waste_sorter.data.pipeline.context import PipelineContext
from yolo_waste_sorter.data.pipeline.runner import Stage
from yolo_waste_sorter.data.remap import manifest_path, remap_sources


def _download_complete(ctx: PipelineContext) -> bool:
    return all((ctx.raw_root / name / MANIFEST_NAME).is_file() for name in ctx.registry)


def _download_run(ctx: PipelineContext) -> str:
    results = download_sources(ctx.registry.values(), ctx.raw_root)
    fetched = sum(1 for result in results if result.action == "fetched")
    return f"{fetched} fetched, {len(results) - fetched} already present -> {ctx.raw_root}"


def download_stage() -> Stage:
    return Stage(
        name="download",
        run=_download_run,
        is_complete=_download_complete,
        hint=(
            "check network access / kaggle CLI auth and the fetcher entries in "
            "configs/datasets.yaml; a checksum mismatch means the registry sha256 "
            "and the archive on disk disagree"
        ),
    )


def _remap_complete(ctx: PipelineContext) -> bool:
    return all(manifest_path(ctx.interim_root, name).is_file() for name in ctx.registry)


def _remap_run(ctx: PipelineContext) -> str:
    manifests = remap_sources(
        ctx.registry.values(), ctx.raw_root, ctx.interim_root, list(ctx.cfg.classes)
    )
    images = sum(sum(manifest.class_counts.values()) for manifest in manifests)
    drops = sum(manifest.drop_count for manifest in manifests)
    errors = sum(len(manifest.errors) for manifest in manifests)
    summary = f"{images} images remapped, {drops} routed to wilderness"
    if errors:
        summary += f", {errors} per-file errors (see {ctx.remapped_root / '.manifests'})"
    return summary


def remap_stage() -> Stage:
    return Stage(
        name="remap",
        run=_remap_run,
        is_complete=_remap_complete,
        hint=(
            "inspect data/raw/<source>/ layout and the source 'mapping' in "
            "configs/datasets.yaml -- the mapping must be total over the class folders"
        ),
    )
