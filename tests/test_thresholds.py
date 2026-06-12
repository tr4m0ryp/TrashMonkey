"""Tests for the T9 rest-bin threshold tuner (task 012). Offline, no GPU."""

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from trashmonkey.models.evaluation.report import ClassEval, EvalReport, TierReport
from trashmonkey.models.thresholds import (
    MAX_WRONG_BIN,
    REST,
    ThresholdError,
    ThresholdParams,
    consensus_decision,
    per_class_tau,
    truth_from_manifest,
    tune_thresholds,
)
from trashmonkey.utils.config import Config, SweepConfig, load_config

CLASSES = ("plastic", "paper", "cardboard", "metal", "glass", "organic")
PARAMS = ThresholdParams(tau_frame=0.40, min_votes=3, high_water=0.60, conf_floor=0.25)


def make_cfg(tau: tuple[float, ...], votes: tuple[int, ...], high: tuple[float, ...]) -> Config:
    base = load_config()
    sweep = SweepConfig(tau_frame=tau, min_votes=votes, high_water=high)
    return dataclasses.replace(
        base, thresholds=dataclasses.replace(base.thresholds, sweep=sweep)
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    return path


def detections_for(object_id: str, class_id: int, score: float, n: int = 3) -> list[dict[str, Any]]:
    """n degraded frames (severities 1..n), one detection each, uniform score."""
    return [
        {
            "image_id": f"{object_id}/img{i}",
            "object_id": object_id,
            "class_id": class_id,
            "score": score,
            "severity": 1 + (i % 3),
        }
        for i in range(n)
    ]


def make_report(conf_at_p95: dict[str, float | None]) -> EvalReport:
    per_class = {
        name: ClassEval(precision=0.9, recall=0.9, map50=0.9, map50_95=0.7, conf_at_p95=value)
        for name, value in conf_at_p95.items()
    }
    tier = TierReport(
        tier="val", split="val", severity=0, map50=0.9, map50_95=0.7,
        overall={}, per_class=per_class, curves_path="curves/val.npz",
    )
    return EvalReport(
        seed=42, best_pt="best.pt", data_yaml="dataset.yaml", classes=CLASSES,
        conf_sweep=0.001, val=tier, test1=tier, test2=(), severity_curve=(),
        escalation={}, detections_path="detections.jsonl",
    )


class TestConsensusDecision:
    def test_wilderness_leak_confident_single_frame_fails_consensus(self) -> None:
        # 0.9 passes ANY single-frame threshold, but one qualified vote < min_votes.
        votes = [(1, 0.9), (0, 0.30), (2, 0.32), (0, 0.35), (5, 0.31)]
        assert consensus_decision(votes, PARAMS) is REST

    def test_strict_majority_edge(self) -> None:
        params = dataclasses.replace(PARAMS, min_votes=2, high_water=0.5)
        tied = [(0, 0.6)] * 3 + [(1, 0.6)] * 3  # 3 of 6 is NOT a strict majority
        assert consensus_decision(tied, params) is REST
        majority = [(0, 0.6)] * 4 + [(1, 0.6)] * 3  # 4 of 7 is
        assert consensus_decision(majority, params) == 0

    def test_high_water_edge(self) -> None:
        votes = [(0, 0.55)] * 5
        assert consensus_decision(votes, PARAMS) is REST  # max 0.55 < 0.60
        assert consensus_decision(votes, dataclasses.replace(PARAMS, high_water=0.55)) == 0

    def test_min_votes_and_empty(self) -> None:
        assert consensus_decision([(0, 0.7)] * 2, PARAMS) is REST  # 2 < min_votes 3
        assert consensus_decision([(0, 0.7)] * 3, PARAMS) == 0
        assert consensus_decision([], PARAMS) is REST
        assert consensus_decision([(0, 0.2)] * 9, PARAMS) is REST  # none qualified

    def test_per_class_tau_mapping(self) -> None:
        params = dataclasses.replace(PARAMS, tau_frame={0: 0.7, 1: 0.4})
        assert consensus_decision([(0, 0.65)] * 5, params) is REST  # below class-0 tau
        assert consensus_decision([(1, 0.65)] * 5, params) == 1
        with pytest.raises(ThresholdError, match="no per-class tau_frame"):
            consensus_decision([(2, 0.65)] * 5, params)


class TestTruthFromManifest:
    def manifest(self, tmp_path: Path, groups: dict[str, str]) -> Path:
        assignments = {
            "plastic/srcA__a.jpg": "val",
            "plastic/srcA__b.jpg": "val",
            "paper/srcB__c.jpg": "val",
            "metal/srcA__e.jpg": "train",
        }
        path = tmp_path / "split_manifest.yaml"
        path.write_text(yaml.safe_dump({"assignments": assignments, "groups": groups}))
        return path

    def test_val_objects_only_grouped(self, tmp_path: Path) -> None:
        groups = {
            "plastic/srcA__a.jpg": "plastic/srcA__a.jpg",
            "plastic/srcA__b.jpg": "plastic/srcA__a.jpg",  # same physical object
            "paper/srcB__c.jpg": "paper/srcB__c.jpg",
            "metal/srcA__e.jpg": "metal/srcA__e.jpg",
        }
        truth = truth_from_manifest(self.manifest(tmp_path, groups), CLASSES)
        assert truth == {"plastic/srcA__a.jpg": 0, "paper/srcB__c.jpg": 1}

    def test_class_conflict_in_group_fails(self, tmp_path: Path) -> None:
        groups = {
            "plastic/srcA__a.jpg": "plastic/srcA__a.jpg",
            "plastic/srcA__b.jpg": "plastic/srcA__a.jpg",
            "paper/srcB__c.jpg": "plastic/srcA__a.jpg",  # paper in a plastic group
            "metal/srcA__e.jpg": "metal/srcA__e.jpg",
        }
        with pytest.raises(ThresholdError, match="mixes classes"):
            truth_from_manifest(self.manifest(tmp_path, groups), CLASSES)


class TestPerClassSwitch:
    SWEEP = SweepConfig(tau_frame=(0.25, 0.4, 0.6), min_votes=(3,), high_water=(0.5,))

    def test_wide_span_triggers_per_class(self) -> None:
        conf = dict.fromkeys(CLASSES, 0.40)
        conf["plastic"], conf["paper"] = 0.30, 0.55  # span 0.25 > 0.1
        conf["glass"] = None  # never sustains p95 -> clamps to sweep max
        anchors = per_class_tau(make_report(conf), self.SWEEP)
        assert anchors is not None
        assert anchors[0] == 0.30 and anchors[1] == 0.55
        assert anchors[4] == 0.6  # None -> max of sweep range
        assert all(0.25 <= tau <= 0.6 for tau in anchors.values())

    def test_narrow_span_stays_global(self) -> None:
        conf = dict.fromkeys(CLASSES, 0.40)
        conf["plastic"] = 0.45  # span 0.05 <= 0.1
        assert per_class_tau(make_report(conf), self.SWEEP) is None

    def test_missing_class_fails(self) -> None:
        conf = {name: 0.4 for name in CLASSES if name != "organic"}
        with pytest.raises(ThresholdError, match="organic"):
            per_class_tau(make_report(conf), self.SWEEP)


@pytest.fixture()
def synthetic(tmp_path: Path) -> tuple[Config, Path, Path, dict[str, int]]:
    """Grid where exactly (tau=0.6, min_votes=3, high_water=0.5) dominates.

    tau=0.4 cells leak both wilderness objects (wrong_bin 2/11 >> 2%);
    high_water=0.7 rejects everything (rest 100%); tau=0.6/hw=0.5 keeps the
    strong objects (score 0.65) and rests the weak (0.55) plus the
    detection-less object: wrong_bin 0, rest 4/9 -- the feasible minimum.
    """
    cfg = make_cfg(tau=(0.4, 0.6), votes=(3,), high=(0.5, 0.7))
    rows: list[dict[str, Any]] = []
    truth: dict[str, int] = {}
    for i in range(5):  # strong: always sorted once qualified
        rows += detections_for(f"strong{i}", class_id=0, score=0.65)
        truth[f"strong{i}"] = 0
    for i in range(3):  # weak: qualified at tau 0.4 only
        rows += detections_for(f"weak{i}", class_id=1, score=0.55)
        truth[f"weak{i}"] = 1
    truth["unseen"] = 3  # no detections at all -> zero votes -> REST
    wild = detections_for("wild0", class_id=2, score=0.55) + detections_for(
        "wild1", class_id=2, score=0.55
    )
    known_jsonl = write_jsonl(tmp_path / "detections.jsonl", rows)
    wild_jsonl = write_jsonl(tmp_path / "wilderness.jsonl", wild)
    return cfg, known_jsonl, wild_jsonl, truth


def test_tuner_recovers_known_optimal_cell(
    synthetic: tuple[Config, Path, Path, dict[str, int]], tmp_path: Path
) -> None:
    cfg, known_jsonl, wild_jsonl, truth = synthetic
    result = tune_thresholds(
        cfg, known_jsonl, truth, tmp_path / "out", wilderness_jsonl=wild_jsonl
    )
    assert (result.params.tau_frame, result.params.min_votes, result.params.high_water) == (
        0.6, 3, 0.5,
    )
    assert result.constraint_met is True
    assert result.wrong_bin_rate == 0.0
    assert result.rest_rate == pytest.approx(4 / 9)
    data = yaml.safe_load(result.thresholds_path.read_text())
    assert data == {
        "tau_frame": 0.6,
        "min_votes": 3,
        "high_water": 0.5,
        "conf_floor": cfg.thresholds.conf_floor,
        "constraint_met": True,
        "selected_metrics": {"wrong_bin_rate": 0.0, "rest_rate": pytest.approx(4 / 9)},
    }


def test_sweep_csv_schema_and_chosen_flag(
    synthetic: tuple[Config, Path, Path, dict[str, int]], tmp_path: Path
) -> None:
    cfg, known_jsonl, wild_jsonl, truth = synthetic
    result = tune_thresholds(
        cfg, known_jsonl, truth, tmp_path / "out", wilderness_jsonl=wild_jsonl
    )
    lines = result.sweep_path.read_text().splitlines()
    assert lines[0] == "tau_frame,min_votes,high_water,wrong_bin_rate,rest_rate,chosen"
    assert len(lines) == 1 + 2 * 1 * 2  # one row per grid cell
    rows = [line.split(",") for line in lines[1:]]
    assert all(len(row) == 6 for row in rows)
    assert sum(int(row[5]) for row in rows) == 1
    chosen = next(row for row in rows if row[5] == "1")
    assert (float(chosen[0]), int(chosen[1]), float(chosen[2])) == (0.6, 3, 0.5)
    leaky = next(row for row in rows if row[0] == "0.4" and row[2] == "0.5")
    assert float(leaky[3]) == pytest.approx(2 / 11) and float(leaky[3]) > MAX_WRONG_BIN


def test_determinism_byte_identical_artifacts(
    synthetic: tuple[Config, Path, Path, dict[str, int]], tmp_path: Path
) -> None:
    cfg, known_jsonl, wild_jsonl, truth = synthetic
    runs = []
    for name in ("run_a", "run_b"):
        result = tune_thresholds(
            cfg, known_jsonl, truth, tmp_path / name, wilderness_jsonl=wild_jsonl
        )
        runs.append((result.thresholds_path.read_bytes(), result.sweep_path.read_bytes()))
    assert runs[0] == runs[1]


def test_constraint_unsatisfiable_falls_back_loudly(tmp_path: Path) -> None:
    cfg = make_cfg(tau=(0.4, 0.6), votes=(3,), high=(0.5, 0.7))
    truth = {"known0": 0, "known1": 0}
    known_jsonl = write_jsonl(
        tmp_path / "d.jsonl",
        detections_for("known0", 0, 0.65) + detections_for("known1", 0, 0.65),
    )
    # 0.95 beats every tau and high_water in the grid -> leaks in EVERY cell.
    wild_jsonl = write_jsonl(tmp_path / "w.jsonl", detections_for("wild", 1, 0.95))
    result = tune_thresholds(cfg, known_jsonl, truth, tmp_path / "out", wilderness_jsonl=wild_jsonl)
    assert result.constraint_met is False
    assert result.wrong_bin_rate == pytest.approx(1 / 3)
    # Lowest wrong_bin everywhere -> tie broken by rest_rate, then grid order.
    assert (result.params.tau_frame, result.params.high_water) == (0.4, 0.5)
    assert yaml.safe_load(result.thresholds_path.read_text())["constraint_met"] is False


def test_per_class_mode_emits_tau_mapping(tmp_path: Path) -> None:
    cfg = make_cfg(tau=(0.4, 0.6), votes=(3,), high=(0.5,))
    truth = {"obj0": 0, "obj1": 1}
    known_jsonl = write_jsonl(
        tmp_path / "d.jsonl",
        detections_for("obj0", 0, 0.65) + detections_for("obj1", 1, 0.65),
    )
    conf = dict.fromkeys(CLASSES, 0.40)
    conf["plastic"], conf["paper"] = 0.30, 0.55  # span > 0.1 -> per-class mode
    result = tune_thresholds(
        cfg, known_jsonl, truth, tmp_path / "out", report=make_report(conf)
    )
    tau = yaml.safe_load(result.thresholds_path.read_text())["tau_frame"]
    assert isinstance(tau, dict) and sorted(tau) == list(range(6))
    assert all(0.4 <= value <= 0.6 for value in tau.values())  # clamped to sweep range
    assert tau[0] < tau[1]  # plastic anchored below paper
    rows = [line.split(",") for line in result.sweep_path.read_text().splitlines()[1:]]
    means = sorted({row[0] for row in rows})
    assert len(means) == 2  # one mean tau per grid value, reported in the tau column


def test_detections_contract_violations_fail_fast(tmp_path: Path) -> None:
    cfg = make_cfg(tau=(0.4,), votes=(3,), high=(0.5,))
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"image_id": "i", "object_id": "o", "class_id": 0, "score": 0.5}\n')
    with pytest.raises(ThresholdError, match="missing field.*severity"):
        tune_thresholds(cfg, bad, {"o": 0}, tmp_path / "out")
    stray = write_jsonl(tmp_path / "stray.jsonl", detections_for("not_in_truth", 0, 0.5))
    with pytest.raises(ThresholdError, match="missing from the truth mapping"):
        tune_thresholds(cfg, stray, {"o": 0}, tmp_path / "out")
