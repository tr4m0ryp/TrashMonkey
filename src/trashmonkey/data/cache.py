"""Drive-backed processed-dataset cache: build once, restore thereafter.

The data pipeline is expensive (download several GB, autobox, dedup, balance).
This module lets a notebook build the processed dataset once, archive it to a
persistent store (Google Drive on Colab), and on every later run restore the
archive instead of rebuilding -- UNLESS the inputs that determine the dataset
changed.

Staleness is decided by a fingerprint over those inputs: the source registry
(``datasets.yaml`` -- sources, caps, mappings, dedup order) plus the config
fields that drive remap/balance/split (classes, seed, eval split). Editing an
unrelated field (e.g. ``train.batch``) does NOT invalidate the cache; changing
a cap or the leave-out source does, so the next run rebuilds automatically.

All functions are pure / filesystem-only and take explicit paths so the
orchestration is testable without Colab or Drive.
"""

from __future__ import annotations

import hashlib
import tarfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from trashmonkey.utils.config import load_config

# Archive members, relative to the data root (mirrors the historical
# `tar -C data processed interim/pipeline`): the processed dataset plus the
# split manifest the evaluate step needs.
DATASET_MEMBERS: tuple[str, ...] = ("processed", "interim/pipeline")


@dataclass(frozen=True)
class CacheDecision:
    """What restore_or_build did. status: local | restored | rebuilt."""

    status: str
    fingerprint: str
    archive: Path


def dataset_fingerprint(config_path: Path, datasets_path: Path) -> str:
    """sha256 of the inputs that determine the processed dataset.

    Covers ``datasets.yaml`` byte-for-byte and the dataset-relevant config
    fields: classes, seed, the split knobs (eval.val_fraction,
    eval.leave_out_source, eval.clean_holdout) and the label-quality filter
    (eval.label_filter). Tuning any of these reshapes the processed pool, so the
    cache must rebuild. eval.escalation/train.* are NOT included -- they affect
    gating/training, not the dataset.
    """
    cfg = load_config(config_path)
    canonical = yaml.safe_dump(
        {
            "classes": list(cfg.classes),
            "seed": cfg.seed,
            "val_fraction": cfg.eval.val_fraction,
            "leave_out_source": cfg.eval.leave_out_source,
            "clean_holdout": {
                "fraction": cfg.eval.clean_holdout.fraction,
                "sources": list(cfg.eval.clean_holdout.sources),
            },
            "label_filter": {
                "min_confidence": cfg.eval.label_filter.min_confidence,
                "max_box_frac": cfg.eval.label_filter.max_box_frac,
                "min_box_frac": cfg.eval.label_filter.min_box_frac,
                "drop_methods": list(cfg.eval.label_filter.drop_methods),
            },
        },
        sort_keys=True,
    ).encode()
    digest = hashlib.sha256()
    digest.update(canonical)
    digest.update(b"\x00")
    digest.update(datasets_path.read_bytes())
    return digest.hexdigest()


def pack_dataset(
    data_root: Path, archive_path: Path, members: Sequence[str] = DATASET_MEMBERS
) -> None:
    """Tar+gzip the dataset members under ``data_root`` into ``archive_path``."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        for member in members:
            src = data_root / member
            if not src.exists():
                raise FileNotFoundError(f"cannot archive missing dataset path: {src}")
            tar.add(src, arcname=member)


def unpack_dataset(archive_path: Path, data_root: Path) -> None:
    """Extract a dataset archive into ``data_root`` (trusted, self-produced)."""
    data_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(data_root, filter="data")


def _read_marker(path: Path) -> str | None:
    return path.read_text().strip() if path.is_file() else None


def _write_marker(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def restore_or_build(
    *,
    config_path: Path,
    datasets_path: Path,
    data_root: Path,
    dataset_yaml: Path,
    archive_path: Path,
    fingerprint_path: Path,
    local_marker: Path,
    build_fn: Callable[[], object],
    force: bool = False,
    save: bool = True,
) -> CacheDecision:
    """Restore the processed dataset from cache, or build it and cache it.

    Order of preference (each gated on a fingerprint match, skipped if
    ``force``):

    1. ``local`` -- the extracted dataset is already on disk from this session.
    2. ``restored`` -- a matching archive exists in the persistent store; unpack
       it (no rebuild). Only when ``save`` is True.
    3. ``rebuilt`` -- nothing valid is cached: call ``build_fn`` (the pipeline),
       then archive it to the persistent store and stamp the fingerprint
       (only when ``save`` is True).

    ``build_fn`` must leave ``dataset_yaml`` on disk; a missing one after a
    build raises. Returns a ``CacheDecision`` describing which path ran.
    """
    fingerprint = dataset_fingerprint(config_path, datasets_path)

    if not force and dataset_yaml.is_file() and _read_marker(local_marker) == fingerprint:
        return CacheDecision("local", fingerprint, archive_path)

    if (
        not force
        and save
        and archive_path.is_file()
        and _read_marker(fingerprint_path) == fingerprint
    ):
        unpack_dataset(archive_path, data_root)
        if not dataset_yaml.is_file():
            raise FileNotFoundError(
                f"restored archive {archive_path} did not contain {dataset_yaml}; "
                "delete the archive and rebuild"
            )
        _write_marker(local_marker, fingerprint)
        return CacheDecision("restored", fingerprint, archive_path)

    build_fn()
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"dataset build finished but {dataset_yaml} is missing")
    if save:
        pack_dataset(data_root, archive_path)
        _write_marker(fingerprint_path, fingerprint)
    _write_marker(local_marker, fingerprint)
    return CacheDecision("rebuilt", fingerprint, archive_path)
