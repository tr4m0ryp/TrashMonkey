"""Threshold-tuner entrypoint: ``python -m trashmonkey.models.thresholds``.

All logic lives in ``trashmonkey.models.thresholding``; this module is
the stable import path -- the Jetson runtime (015) does
``from trashmonkey.models.thresholds import REST, ThresholdParams,
consensus_decision`` -- and the CLI entry that emits the deployment artifact
``thresholds.yaml`` plus ``sweep.csv`` for the plot stage (014).
"""

from __future__ import annotations

from pathlib import Path

from trashmonkey.models.thresholding import (
    MAX_WRONG_BIN,
    REST,
    Decision,
    RestType,
    ThresholdError,
    ThresholdParams,
    TuneResult,
    Vote,
    consensus_decision,
    per_class_tau,
    truth_from_manifest,
    tune_thresholds,
)

__all__ = [
    "MAX_WRONG_BIN",
    "REST",
    "Decision",
    "RestType",
    "ThresholdError",
    "ThresholdParams",
    "TuneResult",
    "Vote",
    "consensus_decision",
    "main",
    "per_class_tau",
    "truth_from_manifest",
    "tune_thresholds",
]


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m trashmonkey.models.thresholds",
        description="T9 rest-bin tuner: consensus sweep -> thresholds.yaml + sweep.csv.",
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="experiment yaml (default: configs/config.yaml)"
    )
    parser.add_argument(
        "--detections", type=Path, required=True, help="VAL detections JSONL (011 dump)"
    )
    parser.add_argument(
        "--manifest", type=Path, required=True, help="split manifest (truth via instance groups)"
    )
    parser.add_argument(
        "--report", type=Path, default=None, help="eval_report.yaml (per-class tau switch)"
    )
    parser.add_argument(
        "--wilderness", type=Path, default=None, help="unknown-object probe JSONL (optional)"
    )
    parser.add_argument("--out", type=Path, required=True, help="artifact output directory")
    args = parser.parse_args(argv)

    from trashmonkey.models.evaluation import load_report
    from trashmonkey.utils.config import load_config

    cfg = load_config(args.config)
    report = None if args.report is None else load_report(args.report)
    truth = truth_from_manifest(args.manifest, cfg.classes)
    result = tune_thresholds(
        cfg,
        args.detections,
        truth,
        args.out,
        wilderness_jsonl=args.wilderness,
        report=report,
    )
    print(f"tau_frame:      {result.params.tau_frame}")
    print(f"min_votes:      {result.params.min_votes}")
    print(f"high_water:     {result.params.high_water}")
    print(f"wrong_bin_rate: {result.wrong_bin_rate:.4f}")
    print(f"rest_rate:      {result.rest_rate:.4f}")
    print(f"constraint_met: {result.constraint_met}")
    print(f"thresholds:     {result.thresholds_path}")
    print(f"sweep:          {result.sweep_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
