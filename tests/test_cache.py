"""Tests for the Drive-backed processed-dataset cache and the build wrapper."""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path

import pytest

from trashmonkey.data import cache
from trashmonkey.data.cache import (
    CacheDecision,
    dataset_fingerprint,
    pack_dataset,
    restore_or_build,
    unpack_dataset,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_CONFIG = REPO_ROOT / "configs" / "config.yaml"
REAL_DATASETS = REPO_ROOT / "configs" / "datasets.yaml"


# --- fingerprint ----------------------------------------------------------------


def _config_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Copy the real config + registry into tmp so they can be mutated."""
    config = tmp_path / "config.yaml"
    datasets = tmp_path / "datasets.yaml"
    shutil.copy(REAL_CONFIG, config)
    shutil.copy(REAL_DATASETS, datasets)
    return config, datasets


def test_fingerprint_is_stable() -> None:
    a = dataset_fingerprint(REAL_CONFIG, REAL_DATASETS)
    b = dataset_fingerprint(REAL_CONFIG, REAL_DATASETS)
    assert a == b and len(a) == 64


def test_fingerprint_changes_when_registry_changes(tmp_path: Path) -> None:
    config, datasets = _config_pair(tmp_path)
    before = dataset_fingerprint(config, datasets)
    datasets.write_text(datasets.read_text() + "\n# a cap edit would land here\n")
    assert dataset_fingerprint(config, datasets) != before


def test_fingerprint_changes_when_eval_split_changes(tmp_path: Path) -> None:
    config, datasets = _config_pair(tmp_path)
    before = dataset_fingerprint(config, datasets)
    config.write_text(config.read_text().replace("leave_out_source: realwaste", "leave_out_source: trashnet"))
    assert dataset_fingerprint(config, datasets) != before


def test_fingerprint_changes_when_clean_holdout_changes(tmp_path: Path) -> None:
    # The clean-holdout fraction reshapes the splits -> must invalidate the cache.
    config, datasets = _config_pair(tmp_path)
    before = dataset_fingerprint(config, datasets)
    config.write_text(config.read_text().replace("fraction: 0.15", "fraction: 0.25"))
    assert dataset_fingerprint(config, datasets) != before


def test_fingerprint_changes_when_label_filter_changes(tmp_path: Path) -> None:
    # The label filter changes which items survive -> must invalidate the cache.
    config, datasets = _config_pair(tmp_path)
    before = dataset_fingerprint(config, datasets)
    config.write_text(config.read_text().replace("min_confidence: 0.30", "min_confidence: 0.50"))
    assert dataset_fingerprint(config, datasets) != before


def test_fingerprint_ignores_unrelated_config_fields(tmp_path: Path) -> None:
    # Changing a training knob must NOT invalidate the dataset cache.
    config, datasets = _config_pair(tmp_path)
    before = dataset_fingerprint(config, datasets)
    config.write_text(config.read_text().replace("batch: 16", "batch: 64"))
    assert dataset_fingerprint(config, datasets) == before


def test_fingerprint_ignores_escalation_floors(tmp_path: Path) -> None:
    # Escalation floors gate model-size selection, not the dataset -> no rebuild.
    config, datasets = _config_pair(tmp_path)
    before = dataset_fingerprint(config, datasets)
    config.write_text(config.read_text().replace("overall_map50: 0.80", "overall_map50: 0.90"))
    assert dataset_fingerprint(config, datasets) == before


# --- pack / unpack round trip ---------------------------------------------------


def _fake_dataset_tree(data_root: Path) -> Path:
    """Create a minimal processed + interim/pipeline tree; return dataset.yaml."""
    dataset_yaml = data_root / "processed" / "baseline" / "dataset.yaml"
    dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
    dataset_yaml.write_text("names: {0: plastic}\n")
    split = data_root / "interim" / "pipeline" / "split.yaml"
    split.parent.mkdir(parents=True, exist_ok=True)
    split.write_text("stage: split\n")
    return dataset_yaml


def test_pack_unpack_round_trip(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _fake_dataset_tree(src)
    archive = tmp_path / "drive" / "processed.tar.gz"
    pack_dataset(src, archive)
    assert archive.is_file()

    dst = tmp_path / "dst"
    unpack_dataset(archive, dst)
    assert (dst / "processed" / "baseline" / "dataset.yaml").read_text() == "names: {0: plastic}\n"
    assert (dst / "interim" / "pipeline" / "split.yaml").is_file()


def test_pack_missing_member_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing dataset path"):
        pack_dataset(tmp_path, tmp_path / "a.tar.gz")


def test_unpack_stays_within_data_root(tmp_path: Path) -> None:
    # The 'data' filter must neutralise any path-traversal member.
    archive = tmp_path / "evil.tar.gz"
    payload = tmp_path / "payload"
    payload.write_text("x")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(payload, arcname="../escaped.txt")
    dst = tmp_path / "dst"
    with pytest.raises(tarfile.OutsideDestinationError):
        unpack_dataset(archive, dst)
    assert not (tmp_path / "escaped.txt").exists()


# --- restore_or_build decision --------------------------------------------------


def _paths(tmp_path: Path) -> dict[str, Path]:
    config, datasets = _config_pair(tmp_path)
    data_root = tmp_path / "data"
    return {
        "config_path": config,
        "datasets_path": datasets,
        "data_root": data_root,
        "dataset_yaml": data_root / "processed" / "baseline" / "dataset.yaml",
        "archive_path": tmp_path / "drive" / "processed.tar.gz",
        "fingerprint_path": tmp_path / "drive" / "processed.fingerprint",
        "local_marker": data_root / ".dataset.fingerprint",
    }


def test_cache_miss_builds_and_saves(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    calls: list[str] = []

    def build() -> None:
        calls.append("built")
        _fake_dataset_tree(paths["data_root"])

    decision = restore_or_build(build_fn=build, **paths)
    assert decision.status == "rebuilt"
    assert calls == ["built"]
    assert paths["archive_path"].is_file()  # saved to the persistent store
    assert paths["fingerprint_path"].read_text() == decision.fingerprint
    assert paths["local_marker"].read_text() == decision.fingerprint


def test_local_hit_skips_build(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    fp = dataset_fingerprint(paths["config_path"], paths["datasets_path"])
    _fake_dataset_tree(paths["data_root"])
    paths["local_marker"].write_text(fp)

    def build() -> None:
        raise AssertionError("build must not run on a warm local cache")

    decision = restore_or_build(build_fn=build, **paths)
    assert decision.status == "local"


def test_drive_hit_restores_without_building(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    fp = dataset_fingerprint(paths["config_path"], paths["datasets_path"])
    # Seed the persistent store from a separate source tree, then wipe local.
    source = tmp_path / "source"
    _fake_dataset_tree(source)
    pack_dataset(source, paths["archive_path"])
    paths["fingerprint_path"].write_text(fp)

    def build() -> None:
        raise AssertionError("build must not run when the archive matches")

    decision = restore_or_build(build_fn=build, **paths)
    assert decision.status == "restored"
    assert paths["dataset_yaml"].is_file()
    assert paths["local_marker"].read_text() == fp


def test_stale_archive_triggers_rebuild(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    source = tmp_path / "source"
    _fake_dataset_tree(source)
    pack_dataset(source, paths["archive_path"])
    paths["fingerprint_path"].write_text("0" * 64)  # mismatched fingerprint
    calls: list[str] = []

    def build() -> None:
        calls.append("built")
        _fake_dataset_tree(paths["data_root"])

    decision = restore_or_build(build_fn=build, **paths)
    assert decision.status == "rebuilt"
    assert calls == ["built"]


def test_force_rebuilds_over_warm_local(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    fp = dataset_fingerprint(paths["config_path"], paths["datasets_path"])
    _fake_dataset_tree(paths["data_root"])
    paths["local_marker"].write_text(fp)
    calls: list[str] = []

    decision = restore_or_build(
        build_fn=lambda: calls.append("built") or _fake_dataset_tree(paths["data_root"]),
        force=True,
        **paths,
    )
    assert decision.status == "rebuilt"
    assert calls == ["built"]


def test_save_false_builds_without_archiving(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    decision = restore_or_build(
        build_fn=lambda: _fake_dataset_tree(paths["data_root"]),
        save=False,
        **paths,
    )
    assert decision.status == "rebuilt"
    assert not paths["archive_path"].exists()  # local-only build, nothing pushed
    assert paths["local_marker"].is_file()


def test_build_that_leaves_no_dataset_yaml_raises(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    with pytest.raises(FileNotFoundError, match="dataset build finished"):
        restore_or_build(build_fn=lambda: None, **paths)


# --- build_dataset wrapper ------------------------------------------------------


def test_build_dataset_threads_ack_review(monkeypatch: pytest.MonkeyPatch) -> None:
    from trashmonkey.data.pipeline import build as build_mod

    seen: dict[str, object] = {}

    class FakeCtx:
        cfg = type("C", (), {"seed": 42})()

    monkeypatch.setattr(
        build_mod,
        "build_context",
        lambda path, *, ack_review, progress=None: seen.update(path=path, ack=ack_review) or FakeCtx(),
    )
    monkeypatch.setattr(build_mod, "set_seed", lambda seed: seen.update(seed=seed))
    monkeypatch.setattr(build_mod, "build_stages", lambda: ("s",))
    monkeypatch.setattr(
        build_mod,
        "run_pipeline",
        lambda stages, ctx, *, start, force, on_stage=None: {"download": "ran"},
    )

    result = build_mod.build_dataset(Path("configs/config.yaml"), ack_review=True)
    assert result == {"download": "ran"}
    assert seen == {"path": Path("configs/config.yaml"), "ack": True, "seed": 42}


def test_cache_module_reexports() -> None:
    assert cache.DATASET_MEMBERS == ("processed", "interim/pipeline")
    assert isinstance(CacheDecision("local", "x", Path(".")), CacheDecision)
