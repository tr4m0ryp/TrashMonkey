"""Append-only experiment log: one JSON line per run (experiments/runs.jsonl).

Every reported number must trace back to a run record (paper-scribe
requirement), so the record carries the resolved config, dataset hash, library
version, device, metrics, the escalation block, and wall-clock time. The
timestamp is injected (``now``) for testability.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trashmonkey.utils.config import Config
from trashmonkey.utils.hashing import sha256_file


def _jsonable(value: object) -> object:
    """Recursively convert config values (Path, tuple) to JSON-safe types."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def resolved_config_dict(cfg: Config) -> dict[str, Any]:
    """Materialize the frozen config as a JSON-safe nested dict."""
    result = _jsonable(dataclasses.asdict(cfg))
    assert isinstance(result, dict)
    return result


def git_commit(repo_root: Path) -> str | None:
    """Current HEAD commit, or None when git is unavailable (still log the run)."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return proc.stdout.strip()


def device_description() -> str:
    """GPU name when CUDA is up, else 'cpu'. Lazy torch import (optional dep)."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return str(torch.cuda.get_device_name(0))
    return "cpu"


def build_run_record(
    *,
    cfg: Config,
    repo_root: Path,
    data_yaml: Path,
    train_kwargs: dict[str, Any],
    metrics: dict[str, Any],
    final_metrics: dict[str, Any] | None,
    escalation: dict[str, Any],
    ultralytics_version: str,
    wall_clock_seconds: float,
    run_dir: Path,
    best_pt: Path,
    smoke: bool,
    resumed_from: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Assemble one runs.jsonl record; ``metrics`` comes from best.pt final eval."""
    timestamp = now if now is not None else datetime.now(timezone.utc)
    kwargs_log = {
        key: [repr(transform) for transform in value] if key == "augmentations" else _jsonable(value)
        for key, value in train_kwargs.items()
    }
    return {
        "timestamp": timestamp.isoformat(),
        "git_commit": git_commit(repo_root),
        "config": resolved_config_dict(cfg),
        "train_kwargs": kwargs_log,
        "dataset_yaml": str(data_yaml),
        "dataset_yaml_sha256": sha256_file(data_yaml),
        "ultralytics_version": ultralytics_version,
        "device": {"requested": train_kwargs.get("device"), "detected": device_description()},
        "metrics": {"best": _jsonable(metrics), "final": _jsonable(final_metrics)},
        "escalation": escalation,
        "wall_clock_seconds": wall_clock_seconds,
        "run_dir": str(run_dir),
        "best_pt": str(best_pt),
        "smoke": smoke,
        "resumed_from": str(resumed_from) if resumed_from is not None else None,
    }


def append_run_record(record: dict[str, Any], runs_jsonl: Path) -> None:
    """Append exactly one JSON line; never rewrite or truncate existing history."""
    runs_jsonl.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True)
    with open(runs_jsonl, "a", encoding="utf-8") as f:
        f.write(line + "\n")
