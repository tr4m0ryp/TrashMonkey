"""train(): the seeded, config-driven yolo11n fine-tune (T7) with the T5 stack.

The custom Albumentations stack is injected through the validated
``train(augmentations=[...])`` hook (ultralytics >= 8.3.226, enforced by
``require_augmentations_support``): it replaces only the trainer's built-in
Albumentations block while every native augmentation stays config-driven.
ultralytics is imported lazily so the package works without it installed.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from trashmonkey.models.training.escalation import check_escalation, extract_metrics
from trashmonkey.models.training.guards import (
    require_augmentations_support,
    validate_train_config,
)
from trashmonkey.models.training.kwargs import SMOKE_OVERRIDES, build_train_kwargs
from trashmonkey.models.training.runlog import append_run_record, build_run_record
from trashmonkey.models.training.smoke import build_smoke_dataset
from trashmonkey.utils.config import Config
from trashmonkey.utils.degrade import build_train_stack
from trashmonkey.utils.seed import set_seed

_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RUNS_JSONL = _REPO_ROOT / "experiments" / "runs.jsonl"
DEFAULT_PROJECT = _REPO_ROOT / "experiments" / "runs"


@dataclass(frozen=True)
class RunResult:
    best_pt: Path
    metrics: dict[str, Any]
    run_dir: Path


def smoke_requested() -> bool:
    return os.environ.get("SMOKE_TEST", "") == "1"


def train(
    cfg: Config,
    data_yaml: Path | None,
    smoke: bool = False,
    *,
    runs_jsonl: Path | None = None,
    project: Path | None = None,
    resume: Path | None = None,
    now: datetime | None = None,
) -> RunResult:
    """Run the T7 fine-tune and append one record to runs.jsonl.

    Args:
        cfg: Typed experiment config (T7 train + T5 augment sections).
        data_yaml: Dataset yaml; may be None only in smoke mode (synthesized).
        smoke: Tiny CPU run through the full cycle (also via SMOKE_TEST=1).
        runs_jsonl: Experiment log path (default experiments/runs.jsonl).
        project: Run-directory parent (default experiments/runs; tmp in smoke).
        resume: last.pt of an interrupted run to continue (epoch, optimizer,
            and EMA state restored by ultralytics; the augmentation stack is
            re-supplied since transforms are not serialized in checkpoints).
            Use ``find_resumable()`` to locate one -- resuming a *finished*
            run raises inside ultralytics.
        now: Injected timestamp for the run record (default: UTC now).
    """
    smoke = smoke or smoke_requested()
    validate_train_config(cfg)
    if data_yaml is None and not smoke:
        raise ValueError("data_yaml is required outside smoke mode")
    if resume is not None:
        if smoke:
            raise ValueError("resume is incompatible with smoke mode (throwaway tmp runs)")
        if not resume.is_file():
            raise FileNotFoundError(f"resume checkpoint missing: {resume}")

    smoke_root: Path | None = None
    if smoke:
        smoke_root = Path(tempfile.mkdtemp(prefix="yolo-waste-smoke-"))
        if data_yaml is None:
            imgsz = int(SMOKE_OVERRIDES["imgsz"])
            data_yaml = build_smoke_dataset(
                cfg.classes, smoke_root / "dataset", imgsz=imgsz, seed=cfg.seed
            )
    assert data_yaml is not None

    set_seed(cfg.seed)
    import ultralytics  # lazy: the package must import without it installed

    require_augmentations_support(ultralytics.__version__)

    kwargs = build_train_kwargs(cfg, data_yaml)
    if smoke:
        kwargs.update(SMOKE_OVERRIDES)
    if project is None:
        project = (smoke_root / "runs") if smoke_root is not None else DEFAULT_PROJECT
    kwargs["project"] = str(project)
    kwargs["name"] = cfg.experiment.name + ("-smoke" if smoke else "")
    kwargs["augmentations"] = build_train_stack(cfg)
    if resume is not None:
        # check_resume restores every arg from the checkpoint (run dir included)
        # and honors only the re-supplied augmentations stack plus a few
        # memory/device knobs; the rest of kwargs serves as the fallback when
        # the checkpoint's stored data path no longer exists on a fresh VM.
        kwargs["resume"] = str(resume)

    model = ultralytics.YOLO(str(resume) if resume is not None else cfg.model.base)
    start = time.perf_counter()
    results = model.train(**kwargs)
    wall_clock_seconds = time.perf_counter() - start
    if results is None:
        raise RuntimeError(
            "model.train() returned no metrics object (DDP rank != 0?); "
            "the run record requires the final DetMetrics"
        )

    trainer = model.trainer
    best_pt = Path(str(trainer.best))
    run_dir = Path(str(trainer.save_dir))
    metrics = extract_metrics(results)
    escalation = check_escalation(metrics, cfg.classes)
    record = build_run_record(
        cfg=cfg,
        repo_root=_REPO_ROOT,
        data_yaml=data_yaml,
        train_kwargs=kwargs,
        metrics=metrics,
        final_metrics=getattr(trainer, "metrics", None),
        escalation=escalation,
        ultralytics_version=str(ultralytics.__version__),
        wall_clock_seconds=wall_clock_seconds,
        run_dir=run_dir,
        best_pt=best_pt,
        smoke=smoke,
        resumed_from=resume,
        now=now,
    )
    append_run_record(record, runs_jsonl if runs_jsonl is not None else DEFAULT_RUNS_JSONL)

    if not best_pt.is_file():
        raise RuntimeError(f"training finished but best.pt is missing: {best_pt}")
    if not metrics["overall"] and not metrics["per_class"]:
        raise RuntimeError("training finished but no metrics could be parsed")
    return RunResult(best_pt=best_pt, metrics=metrics, run_dir=run_dir)
