"""Shared pipeline context: config, source registry, derived roots, stage manifests.

Stage manifests live under data/interim/pipeline/<stage>.yaml and drive the
resume logic: a stage whose manifest (or module-level completion marker)
exists is skipped unless --force.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from trashmonkey.data.autobox import DinoPredictFn, MaskFn
from trashmonkey.data.download.registry import SourceSpec
from trashmonkey.data.remap import REMAPPED_DIRNAME, WILDERNESS_DIRNAME
from trashmonkey.utils.config.schema import Config

PIPELINE_DIRNAME = "pipeline"
AUTOBOX_DIRNAME = "autobox"
QA_DIRNAME = "qa"
REVIEW_DIRNAME = "review"


class StageError(Exception):
    """A pipeline stage failed; the message names the stage and a remedy hint."""


class PipelineHalt(Exception):
    """The QA gate halted the pipeline pending human review (rerun with --ack-review)."""


@dataclass(frozen=True)
class PipelineContext:
    """Everything a stage needs: config, registry, derived paths, gate flags.

    `dino_predict` / `birefnet_mask` are test injection points for the autobox
    chain backends; None means the chain builds the real lazy backends.
    """

    cfg: Config
    registry: dict[str, SourceSpec]
    ack_review: bool = False
    dino_predict: DinoPredictFn | None = None
    birefnet_mask: MaskFn | None = None

    @property
    def raw_root(self) -> Path:
        return self.cfg.paths.raw

    @property
    def interim_root(self) -> Path:
        return self.cfg.paths.interim

    @property
    def processed_root(self) -> Path:
        return self.cfg.paths.processed

    @property
    def remapped_root(self) -> Path:
        return self.interim_root / REMAPPED_DIRNAME

    @property
    def wilderness_root(self) -> Path:
        return self.interim_root / WILDERNESS_DIRNAME

    @property
    def autobox_root(self) -> Path:
        return self.interim_root / AUTOBOX_DIRNAME

    @property
    def qa_root(self) -> Path:
        return self.interim_root / QA_DIRNAME

    @property
    def review_root(self) -> Path:
        return self.interim_root / REVIEW_DIRNAME

    def manifest_path(self, stage: str) -> Path:
        return self.interim_root / PIPELINE_DIRNAME / f"{stage}.yaml"

    def write_manifest(self, stage: str, payload: dict[str, Any]) -> Path:
        path = self.manifest_path(stage)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump({"stage": stage, **payload}, sort_keys=False))
        return path

    def read_manifest(self, stage: str) -> dict[str, Any]:
        path = self.manifest_path(stage)
        if not path.is_file():
            raise StageError(
                f"missing '{stage}' stage manifest at {path} -- run the pipeline "
                f"without --stage (or with --stage {stage}) first"
            )
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict):
            raise StageError(f"malformed stage manifest at {path}: expected a mapping")
        return raw


def manifest_str_list(manifest: dict[str, Any], key: str, stage: str) -> list[str]:
    """Validated list-of-strings field from a stage manifest."""
    value = manifest.get(key)
    if not isinstance(value, list):
        raise StageError(f"'{stage}' manifest: '{key}' must be a list, got {type(value).__name__}")
    out: list[str] = []
    for entry in value:
        if not isinstance(entry, str):
            raise StageError(f"'{stage}' manifest: '{key}' entries must be strings")
        out.append(entry)
    return out
