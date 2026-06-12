"""thresholds.yaml deployment-artifact tests: the fail-fast reader in
``deploy.artifacts`` stays in lockstep with the T9 writer's schema.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from trashmonkey.deploy import load_threshold_params
from trashmonkey.models.thresholding import write_thresholds_yaml
from trashmonkey.models.thresholding.tuner import SweepCell
from trashmonkey.models.thresholds import ThresholdError


def test_thresholds_roundtrip_with_real_writer(tmp_path: Path) -> None:
    cell = SweepCell(
        tau_frame=0.45, tau_mean=0.45, min_votes=3, high_water=0.6,
        wrong_bin_rate=0.01, rest_rate=0.2,
    )
    path = tmp_path / "thresholds.yaml"
    write_thresholds_yaml(cell, conf_floor=0.25, constraint_met=True, path=path)
    params = load_threshold_params(path)
    assert (params.tau_frame, params.min_votes) == (0.45, 3)
    assert (params.high_water, params.conf_floor) == (0.6, 0.25)


def test_thresholds_per_class_mapping(tmp_path: Path) -> None:
    path = tmp_path / "thresholds.yaml"
    path.write_text(
        yaml.safe_dump(
            {"tau_frame": {0: 0.4, 1: 0.55}, "min_votes": 3, "high_water": 0.6,
             "conf_floor": 0.25}
        )
    )
    params = load_threshold_params(path)
    assert params.tau_for(1) == 0.55
    with pytest.raises(ThresholdError, match="no per-class tau_frame entry"):
        params.tau_for(5)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"min_votes": None}, "missing required"),  # None => key deleted below
        ({"surprise": 1}, "unknown key"),
        ({"min_votes": 0}, "min_votes must be an int >= 1"),
        ({"min_votes": True}, "min_votes must be an int >= 1"),
        ({"tau_frame": "high"}, "expected a number"),
        ({"tau_frame": {}}, "must not be empty"),
        ({"tau_frame": {"a": 0.4}}, "class ids must be ints"),
        ({"high_water": [0.6]}, "expected a number"),
    ],
)
def test_thresholds_malformed_raises(tmp_path: Path, mutation: dict, match: str) -> None:
    data: dict = {"tau_frame": 0.4, "min_votes": 3, "high_water": 0.6, "conf_floor": 0.25}
    for key, value in mutation.items():
        if value is None:
            del data[key]
        else:
            data[key] = value
    path = tmp_path / "thresholds.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ThresholdError, match=match):
        load_threshold_params(path)


def test_thresholds_not_a_mapping_and_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "thresholds.yaml"
    path.write_text("- 1\n- 2\n")
    with pytest.raises(ThresholdError, match="top level must be a mapping"):
        load_threshold_params(path)
    with pytest.raises(ThresholdError, match="artifact not found"):
        load_threshold_params(tmp_path / "absent.yaml")
