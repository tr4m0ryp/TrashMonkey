"""Command-line wrapper: python -m yolo_waste_sorter.models.train [--smoke]."""

from __future__ import annotations

from pathlib import Path

from yolo_waste_sorter.models.training.core import train
from yolo_waste_sorter.utils.config import load_config


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m yolo_waste_sorter.models.train",
        description="Seeded, config-driven yolo11n fine-tune (T7 recipe, T5 stack).",
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="experiment yaml (default: configs/config.yaml)"
    )
    parser.add_argument(
        "--data", type=Path, default=None, help="dataset yaml (synthesized in smoke mode if omitted)"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="tiny CPU run through the full cycle (SMOKE_TEST=1 also works)"
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    result = train(cfg, args.data, smoke=args.smoke)
    print(f"best.pt:  {result.best_pt}")
    print(f"run dir:  {result.run_dir}")
    print(f"mAP50:    {result.metrics['map50']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
