"""IoU math against hand-computed boxes, and the directory-level cross-check."""

from pathlib import Path

import pytest
from helpers import write_labels

from yolo_waste_sorter.data.qa import Box, iou_crosscheck, iou_cxcywh


def test_iou_identical_boxes_is_one() -> None:
    box = Box(0, 0.5, 0.5, 0.4, 0.4)
    assert iou_cxcywh(box, box) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero() -> None:
    assert iou_cxcywh(Box(0, 0.2, 0.2, 0.2, 0.2), Box(0, 0.8, 0.8, 0.2, 0.2)) == 0.0


def test_iou_partial_overlap_hand_computed() -> None:
    # A=(0.3,0.3,0.7,0.7), B=(0.4,0.4,0.8,0.8) xyxy: inter 0.3*0.3=0.09,
    # union 0.16+0.16-0.09=0.23 -> IoU = 9/23.
    a, b = Box(0, 0.5, 0.5, 0.4, 0.4), Box(0, 0.6, 0.6, 0.4, 0.4)
    assert iou_cxcywh(a, b) == pytest.approx(9 / 23)


def test_iou_cross_shape_hand_computed() -> None:
    # A=(0.25,0.375,0.75,0.625), B=(0.375,0.25,0.625,0.75): inter 0.25*0.25=0.0625,
    # union 0.125+0.125-0.0625=0.1875 -> IoU = 1/3.
    a, b = Box(0, 0.5, 0.5, 0.5, 0.25), Box(0, 0.5, 0.5, 0.25, 0.5)
    assert iou_cxcywh(a, b) == pytest.approx(1 / 3)


def test_iou_crosscheck_stats(tmp_path: Path) -> None:
    ours, ref = tmp_path / "ours", tmp_path / "ref"
    write_labels(ours, "a", [(0, 0.5, 0.5, 0.4, 0.4)])  # identical -> 1.0
    write_labels(ref, "a", [(0, 0.5, 0.5, 0.4, 0.4)])
    write_labels(ours, "b", [(0, 0.5, 0.5, 0.5, 0.25)])  # cross shape -> 1/3
    write_labels(ref, "b", [(0, 0.5, 0.5, 0.25, 0.5)])
    write_labels(ours, "c", [(0, 0.2, 0.2, 0.2, 0.2)])  # disjoint -> 0.0
    write_labels(ref, "c", [(0, 0.8, 0.8, 0.2, 0.2)])
    write_labels(ours, "only_ours", [(0, 0.5, 0.5, 0.4, 0.4)])  # unpaired: ignored
    stats = iou_crosscheck(ours, ref)
    assert (stats.n_ours, stats.n_reference, stats.n_paired) == (4, 3, 3)
    assert stats.median == pytest.approx(1 / 3)
    assert stats.q1 == pytest.approx(1 / 6)
    assert stats.q3 == pytest.approx(2 / 3)
    assert stats.frac_geq_target == pytest.approx(1 / 3)
    assert stats.per_image["a"] == pytest.approx(1.0)


def test_iou_crosscheck_no_common_stems_raises(tmp_path: Path) -> None:
    ours, ref = tmp_path / "ours", tmp_path / "ref"
    write_labels(ours, "a", [(0, 0.5, 0.5, 0.4, 0.4)])
    write_labels(ref, "b", [(0, 0.5, 0.5, 0.4, 0.4)])
    with pytest.raises(ValueError, match="no filename stems shared"):
        iou_crosscheck(ours, ref)
