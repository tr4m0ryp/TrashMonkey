"""Smoke workdir materialization: config pair, fixture registry, pipeline context.

The smoke profile (configs/smoke.yaml) and the fixture registry
(tests/fixtures/smoke/datasets.yaml) are templates: ``materialize`` rewrites
the config ``paths`` into a throwaway workdir and resolves the registry's
repo-relative archive refs to absolute paths, then builds a PipelineContext
with ack_review=True (the QA gate's T3 acceptance metrics are human-review
-only, hence always pending in-pipeline) and the FORCED-centerbox autobox
backends injected (no detector hits, empty mask -> center box, no downloads).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml
from PIL import Image

from trashmonkey.data.autobox import Detection
from trashmonkey.data.pipeline import PipelineContext, build_context

REPO_ROOT = Path(__file__).resolve().parents[3]
SMOKE_CONFIG = REPO_ROOT / "configs" / "smoke.yaml"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "smoke"
REGEN_HINT = "PYTHONPATH=src python -m trashmonkey.smoke --regen-fixtures"

_DATA_KEYS = ("raw", "interim", "processed", "external")


class SmokeSetupError(Exception):
    """The smoke profile, fixture registry, or fixture archives are unusable."""


def no_detections(image_path: Path) -> Sequence[Detection]:
    """Injected DINO backend: never detects, so the chain falls through."""
    return ()


def empty_mask(image_path: Path) -> npt.NDArray[np.uint8]:
    """Injected BiRefNet backend: all-zero mask -> the chain emits a center box."""
    with Image.open(image_path) as img:
        width, height = img.size
    return np.zeros((height, width), dtype=np.uint8)


def _load_yaml(path: Path, what: str) -> dict[str, Any]:
    if not path.is_file():
        raise SmokeSetupError(f"{what} not found: {path}")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise SmokeSetupError(f"{path}: top level must be a mapping")
    return raw


def _rewrite_paths(raw_cfg: dict[str, Any], workdir: Path) -> dict[str, Any]:
    raw_cfg["paths"] = {key: str(workdir / "data" / key) for key in _DATA_KEYS} | {
        "models": str(workdir / "models"),
        "reports": str(workdir / "reports"),
    }
    return raw_cfg


def _resolve_registry(raw: dict[str, Any]) -> dict[str, Any]:
    sources = raw.get("sources")
    if not isinstance(sources, list):
        raise SmokeSetupError(f"{FIXTURES_DIR / 'datasets.yaml'}: 'sources' must be a list")
    for entry in sources:
        fetcher = entry.get("fetcher", {})
        archive = REPO_ROOT / str(fetcher.get("ref", ""))
        if not archive.is_file():
            raise SmokeSetupError(
                f"fixture archive missing: {archive} -- regenerate with: {REGEN_HINT}"
            )
        fetcher["ref"] = str(archive)
    return raw


def materialize(workdir: Path) -> PipelineContext:
    """Write the workdir config/datasets pair and build the smoke context."""
    raw_cfg = _rewrite_paths(_load_yaml(SMOKE_CONFIG, "smoke config"), workdir)
    registry = _resolve_registry(_load_yaml(FIXTURES_DIR / "datasets.yaml", "fixture registry"))

    cfg_dir = workdir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    config_path = cfg_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw_cfg, sort_keys=False))
    (cfg_dir / "datasets.yaml").write_text(yaml.safe_dump(registry, sort_keys=False))

    ctx = build_context(config_path, ack_review=True)
    return dataclasses.replace(ctx, dino_predict=no_detections, birefnet_mask=empty_mask)
