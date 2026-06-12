"""Training entrypoint module: ``python -m trashmonkey.models.train``.

All logic lives in ``trashmonkey.models.training``; this module is the
stable import path (``from trashmonkey.models.train import train``) and
the CLI entry.
"""

from trashmonkey.models.training import RunResult, train
from trashmonkey.models.training.cli import main

__all__ = ["RunResult", "main", "train"]

if __name__ == "__main__":
    raise SystemExit(main())
