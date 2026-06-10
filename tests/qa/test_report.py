"""Acceptance constants, acceptance_pass logic, and YAML serialization."""

from pathlib import Path

import pytest
import yaml
from helpers import prov, write_labels

from yolo_waste_sorter.data.qa import (
    FLAG_AREA_EXTREME,
    LOC_FAIL_MAX,
    REVIEW_FAIL_MAX,
    TARGET_MEDIAN_IOU,
    QAReport,
    run_checks,
)


def one_clean_image_report(tmp: Path) -> QAReport:
    labels = tmp / "labels"
    write_labels(labels, "img", [(0, 0.5, 0.5, 0.4, 0.4)])
    return run_checks(labels, {"img": prov(tmp, "img")})


def test_acceptance_constants_pin_t3_contract() -> None:
    assert (REVIEW_FAIL_MAX, LOC_FAIL_MAX, TARGET_MEDIAN_IOU) == (0.10, 0.20, 0.80)


def test_acceptance_pass_requires_all_metrics(tmp_path: Path) -> None:
    report = one_clean_image_report(tmp_path)
    with pytest.raises(ValueError, match="incomplete"):
        _ = report.acceptance_pass


def test_acceptance_pass_boundaries(tmp_path: Path) -> None:
    report = one_clean_image_report(tmp_path)
    report.review_fail_rate, report.loc_fail_rate, report.median_iou = 0.10, 0.20, 0.80
    assert report.acceptance_pass  # all bars met exactly at threshold
    for attr, bad in (("review_fail_rate", 0.11), ("loc_fail_rate", 0.21), ("median_iou", 0.79)):
        failing = one_clean_image_report(tmp_path / attr)
        failing.review_fail_rate, failing.loc_fail_rate, failing.median_iou = 0.05, 0.10, 0.90
        setattr(failing, attr, bad)
        assert not failing.acceptance_pass


def test_report_yaml_round_trip(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    write_labels(labels, "flagged", [(0, 0.5, 0.5, 0.02, 0.02)])
    write_labels(labels, "clean", [(0, 0.5, 0.5, 0.4, 0.4)])
    records = {s: prov(tmp_path, s) for s in ("flagged", "clean")}
    report = run_checks(labels, records)
    out = tmp_path / "qa_report.yaml"
    report.to_yaml(out)
    loaded = yaml.safe_load(out.read_text())
    assert loaded["total_images"] == 2 and loaded["flagged_images"] == 1
    assert loaded["per_image_flags"]["flagged"] == [FLAG_AREA_EXTREME]
    assert loaded["per_image_flags"]["clean"] == []
    assert loaded["flag_counts"] == {FLAG_AREA_EXTREME: 1}
    assert loaded["acceptance"]["pass"] is None  # metrics not yet recorded
    assert loaded["acceptance"]["target_median_iou"] == TARGET_MEDIAN_IOU
