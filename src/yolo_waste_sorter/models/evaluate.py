"""Evaluation entrypoint module: ``python -m yolo_waste_sorter.models.evaluate``.

All logic lives in ``yolo_waste_sorter.models.evaluation``; this module is the
stable import path (``from yolo_waste_sorter.models.evaluate import evaluate``)
and the CLI entry.
"""

from __future__ import annotations

from pathlib import Path

from yolo_waste_sorter.models.evaluation import EvalReport, evaluate, load_report

__all__ = ["EvalReport", "evaluate", "load_report", "main"]


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m yolo_waste_sorter.models.evaluate",
        description="T6 three-tier evaluation: VAL, TEST-1, TEST-2 severity curve.",
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="experiment yaml (default: configs/config.yaml)"
    )
    parser.add_argument("--best", type=Path, required=True, help="trained best.pt checkpoint")
    parser.add_argument("--data", type=Path, required=True, help="emitted dataset yaml")
    parser.add_argument(
        "--manifest", type=Path, required=True, help="split manifest with instance-group ids"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="report dir (default: <run>/evaluation)"
    )
    args = parser.parse_args(argv)

    from yolo_waste_sorter.utils.config import load_config

    cfg = load_config(args.config)
    report = evaluate(cfg, args.best, args.data, args.manifest, out_dir=args.out)
    print(f"VAL    mAP50: {report.val.map50:.4f}")
    print(f"TEST-1 mAP50: {report.test1.map50:.4f}")
    for point in report.severity_curve:
        print(f"TEST-2 s{point.severity} mAP50: {point.map50:.4f}")
    print(f"escalation passed: {report.escalation['passed']}")
    print(f"detections: {report.detections_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
