"""Positive + negative tests for every automated check, plus fail-fast input handling."""

from pathlib import Path

import pytest
from helpers import BoxTuple, aspect_box, check_many, check_one, prov, square, write_labels

from trashmonkey.data.qa import (
    FLAG_AREA_EXTREME,
    FLAG_AREA_ZSCORE,
    FLAG_ASPECT_ZSCORE,
    FLAG_BOX_COUNT,
    FLAG_CENTERBOX,
    FLAG_EDGE_CONTACT,
    FLAG_LOW_CONFIDENCE,
    ProvenanceRecord,
    load_provenance,
    run_checks,
)


def test_box_count_zero_boxes_flagged(tmp_path: Path) -> None:
    assert FLAG_BOX_COUNT in check_one(tmp_path, [])


def test_box_count_missing_label_file_flagged(tmp_path: Path) -> None:
    (tmp_path / "labels").mkdir()
    report = run_checks(tmp_path / "labels", {"img": prov(tmp_path, "img")})
    assert FLAG_BOX_COUNT in report.images["img"].flags


def test_box_count_multiple_boxes_flagged(tmp_path: Path) -> None:
    two = [(0, 0.3, 0.3, 0.2, 0.2), (0, 0.7, 0.7, 0.2, 0.2)]
    assert FLAG_BOX_COUNT in check_one(tmp_path, two)


def test_box_count_single_box_not_flagged(tmp_path: Path) -> None:
    assert FLAG_BOX_COUNT not in check_one(tmp_path, [(0, 0.5, 0.5, 0.4, 0.4)])


def test_area_extreme_small_flagged(tmp_path: Path) -> None:
    assert FLAG_AREA_EXTREME in check_one(tmp_path, [(0, 0.5, 0.5, 0.2, 0.2)])  # 4% < 5%


def test_area_extreme_large_flagged(tmp_path: Path) -> None:
    assert FLAG_AREA_EXTREME in check_one(tmp_path, [(0, 0.5, 0.5, 0.98, 0.98)])  # 96% > 95%


def test_area_extreme_mid_not_flagged(tmp_path: Path) -> None:
    assert FLAG_AREA_EXTREME not in check_one(tmp_path, [(0, 0.5, 0.5, 0.5, 0.5)])  # 25%


def test_area_zscore_outlier_flagged_inliers_not(tmp_path: Path) -> None:
    # All square (aspect exactly 1.0 -> zero aspect spread). Areas: nine 0.20,
    # one 0.21, one 0.90 -> hand-computed |z| = 3.16 for 0.90, <= 0.33 otherwise.
    boxes: dict[str, BoxTuple] = {}
    for i in range(9):
        boxes[f"in{i}"] = (0, 0.5, 0.5, *square(0.20))
    boxes["near"] = (0, 0.5, 0.5, *square(0.21))
    boxes["outlier"] = (0, 0.5, 0.5, *square(0.90))
    report = check_many(tmp_path, boxes)
    assert FLAG_AREA_ZSCORE in report.images["outlier"].flags
    assert FLAG_AREA_EXTREME not in report.images["outlier"].flags  # 0.90 < 0.95: isolates zscore
    for stem in [*[f"in{i}" for i in range(9)], "near"]:
        assert FLAG_AREA_ZSCORE not in report.images[stem].flags
        assert FLAG_ASPECT_ZSCORE not in report.images[stem].flags  # zero-variance guard


def test_aspect_zscore_outlier_flagged_inliers_not(tmp_path: Path) -> None:
    # Aspects: nine 1.0, one 1.05, one 5.0 -> hand-computed |z| = 3.16 for 5.0.
    # Areas vary benignly (0.20/0.24/0.18, max |z| = 1.63) so no area flags fire.
    boxes: dict[str, BoxTuple] = {}
    for i in range(9):
        area = 0.20 if i < 5 else 0.24
        boxes[f"in{i}"] = (0, 0.5, 0.5, *aspect_box(area, 1.0))
    boxes["near"] = (0, 0.5, 0.5, *aspect_box(0.24, 1.05))
    boxes["outlier"] = (0, 0.5, 0.5, *aspect_box(0.18, 5.0))
    report = check_many(tmp_path, boxes)
    assert FLAG_ASPECT_ZSCORE in report.images["outlier"].flags
    for stem in [*[f"in{i}" for i in range(9)], "near"]:
        assert FLAG_ASPECT_ZSCORE not in report.images[stem].flags
    for rec in report.images.values():
        assert FLAG_AREA_ZSCORE not in rec.flags


def test_edge_contact_three_edges_flagged(tmp_path: Path) -> None:
    assert FLAG_EDGE_CONTACT in check_one(tmp_path, [(0, 0.5, 0.7, 1.0, 0.6)])  # L, R, bottom


def test_edge_contact_two_edges_not_flagged(tmp_path: Path) -> None:
    assert FLAG_EDGE_CONTACT not in check_one(tmp_path, [(0, 0.5, 0.5, 1.0, 0.5)])  # L, R only


def test_low_confidence_flagged(tmp_path: Path) -> None:
    assert FLAG_LOW_CONFIDENCE in check_one(tmp_path, [(0, 0.5, 0.5, 0.4, 0.4)], confidence=0.29)


def test_confidence_at_threshold_not_flagged(tmp_path: Path) -> None:
    assert FLAG_LOW_CONFIDENCE not in check_one(
        tmp_path, [(0, 0.5, 0.5, 0.4, 0.4)], confidence=0.30
    )


def test_centerbox_always_flagged(tmp_path: Path) -> None:
    flags = check_one(tmp_path, [(0, 0.5, 0.5, 0.4, 0.4)], method="centerbox", confidence=0.99)
    assert FLAG_CENTERBOX in flags


def test_dino_and_birefnet_not_centerbox_flagged(tmp_path: Path) -> None:
    for method in ("dino", "birefnet"):
        flags = check_one(tmp_path, [(0, 0.5, 0.5, 0.4, 0.4)], method=method)
        assert FLAG_CENTERBOX not in flags


def test_provenance_jsonl_roundtrip(tmp_path: Path) -> None:
    jsonl = tmp_path / "provenance.jsonl"
    jsonl.write_text(
        '{"image": "a.jpg", "source": "s", "method": "dino", "confidence": 0.8, "flags": []}\n'
        '{"image": "b.jpg", "source": "s", "method": "centerbox", "confidence": 0.0,'
        ' "flags": ["fallback"]}\n'
    )
    records = load_provenance(jsonl)
    assert records["a"].method == "dino" and records["b"].flags == ("fallback",)


def test_unknown_method_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown autobox method"):
        ProvenanceRecord("a.jpg", "s", "yolo", 0.5)


def test_stray_label_without_provenance_rejected(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    write_labels(labels, "img", [(0, 0.5, 0.5, 0.4, 0.4)])
    write_labels(labels, "stray", [(0, 0.5, 0.5, 0.4, 0.4)])
    with pytest.raises(ValueError, match="without provenance"):
        run_checks(labels, {"img": prov(tmp_path, "img")})


def test_malformed_label_line_rejected(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "img.txt").write_text("0 0.5 0.5 1.5 0.5\n")  # w outside [0, 1]
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        run_checks(labels, {"img": prov(tmp_path, "img")})
