"""Tests for interrupted-run detection (resume-from-last.pt). torch is mocked."""

import json
import os
import sys
import types
from pathlib import Path

import pytest

from trashmonkey.models.training import checkpoint_epoch, find_resumable


@pytest.fixture(autouse=True)
def _fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """torch.load stand-in: checkpoints are JSON files; 'corrupt' raises."""
    module = types.ModuleType("torch")

    def load(path: object, map_location: object = None, weights_only: bool = True) -> object:
        text = Path(str(path)).read_text()
        if text == "corrupt":
            raise RuntimeError("invalid load key")
        return json.loads(text)

    module.load = load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", module)


def _make_run(runs_root: Path, name: str, content: str, mtime: float) -> Path:
    weights = runs_root / name / "weights"
    weights.mkdir(parents=True)
    last_pt = weights / "last.pt"
    last_pt.write_text(content)
    os.utime(last_pt, (mtime, mtime))
    return last_pt


def _interrupted(epoch: int = 41) -> str:
    return json.dumps({"epoch": epoch})


_FINISHED = json.dumps({"epoch": -1})  # strip_optimizer marks completion


# --- checkpoint_epoch -------------------------------------------------------------


def test_epoch_read_from_checkpoint(tmp_path: Path) -> None:
    last_pt = _make_run(tmp_path, "run", _interrupted(7), mtime=1000.0)
    assert checkpoint_epoch(last_pt) == 7


def test_epoch_none_when_torch_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)  # forces ImportError
    last_pt = _make_run(tmp_path, "run", _interrupted(), mtime=1000.0)
    assert checkpoint_epoch(last_pt) is None


def test_epoch_none_on_corrupt_checkpoint(tmp_path: Path) -> None:
    last_pt = _make_run(tmp_path, "run", "corrupt", mtime=1000.0)
    assert checkpoint_epoch(last_pt) is None


@pytest.mark.parametrize("content", ["[1, 2]", "{}"])
def test_epoch_none_without_epoch_field(tmp_path: Path, content: str) -> None:
    last_pt = _make_run(tmp_path, "run", content, mtime=1000.0)
    assert checkpoint_epoch(last_pt) is None


# --- find_resumable ----------------------------------------------------------------


def test_missing_root_returns_none(tmp_path: Path) -> None:
    assert find_resumable(tmp_path / "absent") is None


def test_finished_run_not_resumable(tmp_path: Path) -> None:
    _make_run(tmp_path, "run-1", _FINISHED, mtime=1000.0)
    assert find_resumable(tmp_path) is None


def test_interrupted_run_found(tmp_path: Path) -> None:
    last_pt = _make_run(tmp_path, "run-1", _interrupted(), mtime=1000.0)
    assert find_resumable(tmp_path) == last_pt


def test_newest_interrupted_wins(tmp_path: Path) -> None:
    _make_run(tmp_path, "run-old", _interrupted(), mtime=1000.0)
    newest = _make_run(tmp_path, "run-new", _interrupted(), mtime=2000.0)
    assert find_resumable(tmp_path) == newest


def test_newest_finished_falls_through_to_older_interrupted(tmp_path: Path) -> None:
    older = _make_run(tmp_path, "run-old", _interrupted(), mtime=1000.0)
    _make_run(tmp_path, "run-new", _FINISHED, mtime=2000.0)
    assert find_resumable(tmp_path) == older


def test_corrupt_checkpoint_skipped(tmp_path: Path) -> None:
    older = _make_run(tmp_path, "run-old", _interrupted(), mtime=1000.0)
    _make_run(tmp_path, "run-new", "corrupt", mtime=2000.0)
    assert find_resumable(tmp_path) == older


def test_rundirs_without_last_pt_ignored(tmp_path: Path) -> None:
    (tmp_path / "run-empty" / "weights").mkdir(parents=True)
    (tmp_path / "stray-file").write_text("not a run dir")
    assert find_resumable(tmp_path) is None
