"""Training entrypoint module: ``python -m yolo_waste_sorter.models.train``.

All logic lives in ``yolo_waste_sorter.models.training``; this module is the
stable import path (``from yolo_waste_sorter.models.train import train``) and
the CLI entry.
"""

from yolo_waste_sorter.models.training import RunResult, train
from yolo_waste_sorter.models.training.cli import main

__all__ = ["RunResult", "main", "train"]

if __name__ == "__main__":
    raise SystemExit(main())
