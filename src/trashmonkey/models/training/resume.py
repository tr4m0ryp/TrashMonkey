"""Interrupted-run detection for resume-from-last.pt (Colab disconnect recovery).

ultralytics marks a *finished* run by stripping last.pt and setting its
``epoch`` field to -1 (``strip_optimizer``); an *interrupted* run keeps
``epoch >= 0``. Requesting resume on a finished run raises inside ultralytics
("nothing to resume"), so the manager notebook must only pass genuinely
interrupted checkpoints to ``train(resume=...)`` -- that filter lives here.
"""

from __future__ import annotations

from pathlib import Path


def checkpoint_epoch(last_pt: Path) -> int | None:
    """The ``epoch`` field stored in a checkpoint, or None when unreadable.

    None covers torch being unavailable (no training possible anyway) and
    corrupt/truncated checkpoints left behind by a hard crash mid-save.
    """
    try:
        import torch  # lazy: only present where training can actually run
    except ImportError:
        return None
    try:
        ckpt = torch.load(last_pt, map_location="cpu", weights_only=False)
    except Exception:  # torch.load failures span pickle/zip/IO error types
        return None
    if not isinstance(ckpt, dict) or "epoch" not in ckpt:
        return None
    return int(ckpt["epoch"])


def find_resumable(runs_root: Path) -> Path | None:
    """Newest interrupted run's last.pt under ``runs_root``, or None.

    Run directories are scanned newest-first (last.pt mtime); the first
    checkpoint with ``epoch >= 0`` wins. Finished runs (``epoch == -1``) and
    unreadable checkpoints are skipped.
    """
    if not runs_root.is_dir():
        return None
    candidates = [
        last_pt
        for run_dir in runs_root.iterdir()
        if (last_pt := run_dir / "weights" / "last.pt").is_file()
    ]
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for last_pt in candidates:
        epoch = checkpoint_epoch(last_pt)
        if epoch is not None and epoch >= 0:
            return last_pt
    return None
