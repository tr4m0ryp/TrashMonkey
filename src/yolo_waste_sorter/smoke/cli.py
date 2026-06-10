"""End-to-end smoke harness CLI (task 016): ``python -m yolo_waste_sorter.smoke``.

Chain: fixture download -> remap -> autobox (forced centerbox) -> qa (acked)
-> dedup -> balance -> split -> 1-epoch CPU train -> three-tier evaluate ->
wilderness dump -> reduced threshold sweep. One ``PASS <step>: ...`` stdout
line per step; the first failure prints ``FAIL <step>: ...`` and exits 1.

Model backend: real ultralytics when importable; ``FAKE_MODEL=1`` forces the
injected fake (offline, no weights). Neither -> a hard error naming both
options -- no silent fallback.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from collections.abc import Callable, Sequence
from importlib.util import find_spec
from pathlib import Path
from typing import TypeVar

from yolo_waste_sorter.data.pipeline import PipelineContext, build_stages
from yolo_waste_sorter.smoke import fixtures, steps
from yolo_waste_sorter.smoke.workspace import FIXTURES_DIR, SMOKE_CONFIG, materialize
from yolo_waste_sorter.utils.seed import set_seed

_T = TypeVar("_T")

MISSING_MODEL_MESSAGE = (
    "FAIL preflight: no model backend available. Either install ultralytics "
    "(pip install 'ultralytics>=8.3.226,<9'; first run downloads the ~5MB "
    "yolo11n.pt, cached under models/) or run with FAKE_MODEL=1 "
    "(make smoke FAKE_MODEL=1) to inject the offline fake-model mocks."
)


class StepFailure(Exception):
    """A smoke step failed; the message is the FAIL line."""

    def __init__(self, step: str, cause: BaseException) -> None:
        super().__init__(f"FAIL {step}: {cause}")
        self.step = step


def _attempt(name: str, fn: Callable[[], _T]) -> _T:
    try:
        return fn()
    except Exception as exc:
        raise StepFailure(name, exc) from exc


def _print_pass(name: str, result: steps.StepResult) -> None:
    print(f"PASS {name}: {result.summary} [artifact: {result.artifact}]")


def _stage_artifact(ctx: PipelineContext, name: str) -> Path:
    if name == "download":
        return ctx.raw_root
    if name == "remap":
        return ctx.remapped_root
    if name == "split":
        return ctx.processed_root / ctx.cfg.experiment.name / "dataset.yaml"
    return ctx.manifest_path(name)


def _run_chain(workdir: Path) -> None:
    ctx = _attempt("setup", lambda: materialize(workdir))
    set_seed(ctx.cfg.seed)
    for stage in build_stages():
        summary = _attempt(stage.name, lambda: stage.run(ctx))
        _print_pass(
            stage.name,
            steps.StepResult(summary=summary, artifact=_stage_artifact(ctx, stage.name)),
        )

    cfg = ctx.cfg
    data_yaml = ctx.processed_root / cfg.experiment.name / "dataset.yaml"
    split_manifest = ctx.manifest_path("split")

    run_result, train_step = _attempt(
        "train", lambda: steps.run_training(cfg, data_yaml, workdir)
    )
    _print_pass("train", train_step)

    report, eval_step = _attempt(
        "evaluate",
        lambda: steps.run_evaluation(cfg, run_result.best_pt, data_yaml, split_manifest, workdir),
    )
    _print_pass("evaluate", eval_step)

    wild_step = _attempt(
        "wilderness",
        lambda: steps.dump_wilderness(cfg, run_result.best_pt, ctx.wilderness_root, workdir),
    )
    _print_pass("wilderness", wild_step)

    _, tune_step = _attempt(
        "thresholds",
        lambda: steps.run_thresholds(cfg, report, split_manifest, wild_step.artifact, workdir),
    )
    _print_pass("thresholds", tune_step)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m yolo_waste_sorter.smoke",
        description=(
            "End-to-end smoke harness: fixture pipeline -> 1-epoch CPU train -> "
            "three-tier evaluate -> threshold sweep. Uses real ultralytics when "
            "importable (first run fetches the ~5MB yolo11n.pt once, cached under "
            "models/); FAKE_MODEL=1 forces the offline fake-model mocks instead."
        ),
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="throwaway working directory (default: a fresh temp dir)",
    )
    parser.add_argument(
        "--regen-fixtures",
        action="store_true",
        help=f"regenerate the synthetic fixture archives under {FIXTURES_DIR} and exit",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if args.regen_fixtures:
        counts = fixtures.generate_fixtures(FIXTURES_DIR)
        print(f"regenerated fixtures under {FIXTURES_DIR}: {counts}")
        return 0

    fake = os.environ.get("FAKE_MODEL", "") == "1"
    if not fake and find_spec("ultralytics") is None:
        print(MISSING_MODEL_MESSAGE, file=sys.stderr)
        return 1

    workdir = Path(tempfile.mkdtemp(prefix="yws-smoke-")) if args.workdir is None else args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"smoke workdir: {workdir} (model backend: {'fake' if fake else 'ultralytics'})")

    try:
        if fake:
            from yolo_waste_sorter.smoke.fakes import install_fake_ultralytics
            from yolo_waste_sorter.utils.config import load_config

            install_fake_ultralytics(load_config(SMOKE_CONFIG).classes)
        _run_chain(workdir)
    except StepFailure as failure:
        print(failure, file=sys.stderr)
        return 1
    print(f"SMOKE OK: all steps passed (workdir: {workdir})")
    return 0
