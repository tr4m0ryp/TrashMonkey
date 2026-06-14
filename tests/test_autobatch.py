"""Tests for the reproducible VRAM -> (batch, workers) resolver (autobatch)."""

from __future__ import annotations

import pytest

from trashmonkey.models.training import RuntimeProfile, apply_runtime
from trashmonkey.models.training.autobatch import _select, detect_runtime
from trashmonkey.utils.config import Config, load_config


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config()


# --- tier table: a given card always resolves to the same recipe ----------------


@pytest.mark.parametrize(
    ("vram_gb", "expected_batch"),
    [
        (80.0, 64),  # A100-80 / H100-80
        (40.0, 64),  # A100-40 (boundary)
        (39.9, 48),  # just below -> L4 tier
        (24.0, 48),  # L4
        (16.0, 32),  # T4 / G4
        (14.0, 32),  # T4 boundary
        (13.9, 16),  # just below -> default
        (0.0, 16),   # CPU fallback
    ],
)
def test_tier_batch_is_deterministic(vram_gb: float, expected_batch: int) -> None:
    batch, _workers = _select(vram_gb, cpu_count=64)
    assert batch == expected_batch


def test_batch_never_exceeds_nominal_batch_size() -> None:
    # batch must stay <= nbs (64) so accumulate=1 and the recipe is preserved.
    for floor, batch, _workers in [(80.0, 64, 16)] + [(v, *_select(v, 64)) for v in (40, 24, 16, 0)]:
        assert batch <= 64, f"vram tier {floor} resolved batch {batch} > nbs"


def test_workers_clamped_to_cpu_count() -> None:
    # The A100 tier wants 16 workers; a 4-core host must not over-subscribe.
    _batch, workers = _select(40.0, cpu_count=4)
    assert workers == 4
    _batch, workers = _select(40.0, cpu_count=32)
    assert workers == 16


# --- apply_runtime: pure replace over the frozen config -------------------------


def test_apply_runtime_replaces_only_batch_and_workers(cfg: Config) -> None:
    profile = RuntimeProfile("cuda:0", "A100-SXM4-40GB", 40.0, batch=64, workers=16)
    resolved, returned = apply_runtime(cfg, profile)
    assert returned is profile
    assert resolved.train.batch == 64
    assert resolved.train.workers == 16
    # Every other train knob is untouched (recipe preserved).
    assert resolved.train.optimizer == cfg.train.optimizer
    assert resolved.train.lr0 == cfg.train.lr0
    assert resolved.train.epochs == cfg.train.epochs
    assert resolved.train.cache == cfg.train.cache
    # The original frozen config is not mutated.
    assert cfg.train.batch == 16


def test_apply_runtime_cpu_fallback_matches_default(cfg: Config) -> None:
    # No CUDA -> the resolver yields the pinned config.yaml default, so a local
    # CPU "Run all" reproduces the baseline recipe unchanged.
    profile = RuntimeProfile("cpu", "cpu", 0.0, batch=16, workers=8)
    resolved, _ = apply_runtime(cfg, profile)
    assert (resolved.train.batch, resolved.train.workers) == (16, 8)


def test_detect_runtime_without_torch_is_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def _no_torch(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch":
            raise ImportError("torch absent in this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_torch)
    profile = detect_runtime()
    assert profile.device == "cpu"
    assert (profile.batch, profile.workers[0] if isinstance(profile.workers, tuple) else profile.workers) == (16, 8)
    assert "batch=16" in profile.summary()
