"""End-to-end smoke harness test (task 016): the FAKE_MODEL=1 path in a tmp dir.

Runs ``python -m trashmonkey.smoke`` as a subprocess (the fake
ultralytics is injected into the child's sys.modules, never this process's)
and asserts the full contract: exit 0, one PASS line per step, the processed
dataset + eval report + thresholds.yaml + sweep.csv on disk, and the
committed fixture payload staying under 1MB.
"""

from __future__ import annotations

import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "smoke"
MAX_FIXTURE_BYTES = 1_000_000

EXPECTED_STEPS = (
    "download",
    "remap",
    "autobox",
    "qa",
    "dedup",
    "balance",
    "split",
    "train",
    "evaluate",
    "wilderness",
    "thresholds",
)


def test_fixture_payload_under_1mb() -> None:
    files = [p for p in FIXTURES_DIR.rglob("*") if p.is_file()]
    assert files, f"no fixture files under {FIXTURES_DIR}"
    total = sum(p.stat().st_size for p in files)
    assert total < MAX_FIXTURE_BYTES, f"fixtures total {total} bytes (cap {MAX_FIXTURE_BYTES})"


@pytest.mark.slow
def test_fake_model_smoke_end_to_end(tmp_path: Path) -> None:
    workdir = tmp_path / "smoke"
    env = os.environ | {
        "FAKE_MODEL": "1",
        "NO_ALBUMENTATIONS_UPDATE": "1",
        "PYTHONPATH": str(REPO_ROOT / "src"),
    }
    proc = subprocess.run(
        [sys.executable, "-m", "trashmonkey.smoke", "--workdir", str(workdir)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, f"smoke failed:\n{proc.stdout}\n{proc.stderr}"

    for step in EXPECTED_STEPS:
        assert f"PASS {step}: " in proc.stdout, f"missing PASS line for {step}:\n{proc.stdout}"
    assert "SMOKE OK" in proc.stdout
    assert "FAIL " not in proc.stdout

    dataset_yaml = workdir / "data" / "processed" / "smoke" / "dataset.yaml"
    assert dataset_yaml.is_file(), f"processed dataset missing: {dataset_yaml}"
    eval_report = workdir / "evaluation" / "eval_report.yaml"
    assert eval_report.is_file(), f"eval report missing: {eval_report}"
    wilderness = workdir / "evaluation" / "wilderness_detections.jsonl"
    assert wilderness.is_file(), f"wilderness dump missing: {wilderness}"
    thresholds = workdir / "thresholds" / "thresholds.yaml"
    assert thresholds.is_file(), f"thresholds.yaml missing: {thresholds}"
    sweep = workdir / "thresholds" / "sweep.csv"
    assert sweep.is_file(), f"sweep.csv missing: {sweep}"


def test_no_backend_exits_with_clear_message(tmp_path: Path) -> None:
    """Without ultralytics and without FAKE_MODEL=1, the preflight names both options."""
    if find_spec("ultralytics") is not None:
        pytest.skip("ultralytics importable here; the no-backend preflight is unreachable")
    env = {
        key: value for key, value in os.environ.items() if key != "FAKE_MODEL"
    } | {"NO_ALBUMENTATIONS_UPDATE": "1", "PYTHONPATH": str(REPO_ROOT / "src")}
    proc = subprocess.run(
        [sys.executable, "-m", "trashmonkey.smoke", "--workdir", str(tmp_path / "w")],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1
    assert "FAIL preflight" in proc.stderr
    assert "ultralytics" in proc.stderr and "FAKE_MODEL=1" in proc.stderr
