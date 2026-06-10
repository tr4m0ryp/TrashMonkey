"""CLI: python -m yolo_waste_sorter.data.pipeline run [--stage] [--force] [--ack-review] [--config]

The source registry is loaded from the datasets.yaml SIBLING of the config
file, so `--config tmp/config.yaml` pairs with `tmp/datasets.yaml` (defaults:
configs/config.yaml + configs/datasets.yaml). Exit codes: 0 success, 1 stage
or config failure, 2 QA gate halt (review, then rerun with --ack-review).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from yolo_waste_sorter.data.download.errors import DownloadError
from yolo_waste_sorter.data.download.registry import load_registry
from yolo_waste_sorter.data.pipeline.context import PipelineContext, PipelineHalt, StageError
from yolo_waste_sorter.data.pipeline.runner import run_pipeline
from yolo_waste_sorter.data.pipeline.stages import build_stages
from yolo_waste_sorter.utils.config import ConfigError, load_config
from yolo_waste_sorter.utils.seed import set_seed

DEFAULT_CONFIG = Path("configs/config.yaml")
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_QA_HALT = 2


def build_context(config_path: Path, *, ack_review: bool = False) -> PipelineContext:
    """Load the typed config plus the sibling datasets.yaml registry."""
    cfg = load_config(config_path)
    registry = load_registry(config_path.parent / "datasets.yaml", cfg.classes)
    return PipelineContext(cfg=cfg, registry=registry, ack_review=ack_review)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m yolo_waste_sorter.data.pipeline",
        description="Resumable data pipeline: download -> remap -> autobox -> qa -> "
        "dedup -> balance -> split",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the pipeline (completed stages are skipped)")
    run.add_argument(
        "--stage", default=None, help="start at this stage (earlier stages must be complete)"
    )
    run.add_argument(
        "--force", action="store_true", help="re-run stages even when already complete"
    )
    run.add_argument(
        "--ack-review",
        action="store_true",
        help="continue past a failed QA gate after human review of the queue",
    )
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        ctx = build_context(args.config, ack_review=args.ack_review)
        set_seed(ctx.cfg.seed)
        run_pipeline(build_stages(), ctx, start=args.stage, force=args.force)
    except PipelineHalt as halt:
        print(f"pipeline halted: {halt}", file=sys.stderr)
        return EXIT_QA_HALT
    except (StageError, ConfigError, DownloadError) as exc:
        print(f"pipeline failed: {exc}", file=sys.stderr)
        return EXIT_FAILED
    return EXIT_OK
