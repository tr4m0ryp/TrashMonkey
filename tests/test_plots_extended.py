"""Figure tests for the tier/training/degradation/open-set plots (task 014)."""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from test_plots import CLASSES, PNG_MAGIC, _tier

from trashmonkey.models.evaluation.report import EvalReport, SeverityPoint
from trashmonkey.visualization.plots import (
    PlotError,
    plot_confidence_separation,
    plot_degradation_grid,
    plot_tier_comparison,
    plot_training_curves,
    read_detections_jsonl,
    read_results_csv,
    top_scores_per_frame,
)

matplotlib.use("Agg", force=True)


def _assert_png(path: Path, min_bytes: int = 5_000) -> None:
    assert path.is_file()
    data = path.read_bytes()
    assert data[:8] == PNG_MAGIC
    assert len(data) > min_bytes
    assert plt.get_fignums() == []


@pytest.fixture()
def report(tmp_path: Path) -> EvalReport:
    return EvalReport(
        seed=42,
        best_pt="weights/best.pt",
        data_yaml="dataset.yaml",
        classes=CLASSES,
        conf_sweep=0.001,
        val=_tier("val", "val", 0, str(tmp_path / "c_val.npz"), 0.95),
        test1=_tier("test1", "test", 0, str(tmp_path / "c_t1.npz"), 0.90),
        test2=tuple(
            _tier(f"test2_s{s}", "test", s, str(tmp_path / f"c_s{s}.npz"), 0.90 - 0.1 * s)
            for s in (1, 3, 5)
        ),
        severity_curve=(SeverityPoint(severity=1, map50=0.8, map50_95=0.6),),
        escalation={"escalate": False},
        detections_path="detections.jsonl",
    )


@pytest.fixture()
def results_csv(tmp_path: Path) -> Path:
    # Space-padded headers on purpose: the loader must strip them.
    header = (
        "epoch, train/box_loss, train/cls_loss, train/dfl_loss,"
        " metrics/precision(B), metrics/recall(B), metrics/mAP50(B),"
        " metrics/mAP50-95(B), val/box_loss, val/cls_loss, val/dfl_loss"
    )
    rows = [header]
    for epoch in range(1, 21):
        decay = 1.0 / epoch
        rise = 1.0 - decay
        rows.append(
            f"{epoch},{1.2 * decay:.4f},{1.5 * decay:.4f},{1.1 * decay:.4f},"
            f"{0.5 + 0.45 * rise:.4f},{0.5 + 0.4 * rise:.4f},{0.5 + 0.45 * rise:.4f},"
            f"{0.3 + 0.45 * rise:.4f},{1.3 * decay:.4f},{1.6 * decay:.4f},{1.2 * decay:.4f}"
        )
    path = tmp_path / "results.csv"
    path.write_text("\n".join(rows) + "\n")
    return path


def _write_detections(path: Path, scores: dict[str, list[float]]) -> Path:
    with open(path, "w") as f:
        for image_id, values in scores.items():
            for score in values:
                line = {
                    "image_id": image_id,
                    "object_id": image_id,
                    "class_id": 0,
                    "score": score,
                    "severity": 0,
                }
                f.write(json.dumps(line) + "\n")
    return path


@pytest.fixture()
def detection_files(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(42)
    known = {
        f"plastic/src__{i}.jpg": [float(s) for s in rng.uniform(0.55, 0.99, size=3)]
        for i in range(30)
    }
    probes = {
        f"wilderness/probe_{i}.jpg": [float(s) for s in rng.uniform(0.05, 0.7, size=2)]
        for i in range(20)
    }
    return (
        _write_detections(tmp_path / "detections.jsonl", known),
        _write_detections(tmp_path / "wilderness_detections.jsonl", probes),
    )


# --- figure rendering -------------------------------------------------------------


def test_tier_comparison(tmp_path: Path, report: EvalReport) -> None:
    out = tmp_path / "tiers.png"
    plot_tier_comparison(report, save_path=out)
    _assert_png(out)


def test_tier_comparison_from_report_path(tmp_path: Path, report: EvalReport) -> None:
    report_path = tmp_path / "eval_report.yaml"
    report.write_yaml(report_path)
    out = tmp_path / "tiers_from_path.png"
    plot_tier_comparison(report_path, save_path=out)
    _assert_png(out)


def test_training_curves(tmp_path: Path, results_csv: Path) -> None:
    out = tmp_path / "training.png"
    plot_training_curves(results_csv, save_path=out)
    _assert_png(out)


def test_degradation_grid(tmp_path: Path) -> None:
    import cv2

    rng = np.random.default_rng(42)
    img = np.full((96, 128, 3), 235, dtype=np.uint8)
    img[28:68, 44:84] = rng.integers(0, 200, size=(40, 40, 3), dtype=np.uint8)
    image_path = tmp_path / "item.jpg"
    assert cv2.imwrite(str(image_path), img)
    out = tmp_path / "degradation.png"
    plot_degradation_grid(image_path, save_path=out, seed=42)
    _assert_png(out)


def test_confidence_separation(tmp_path: Path, detection_files: tuple[Path, Path]) -> None:
    detections, wilderness = detection_files
    out = tmp_path / "openset.png"
    plot_confidence_separation(detections, wilderness, save_path=out, tau_frame=0.40)
    _assert_png(out)


# --- parsing and contracts ---------------------------------------------------------


def test_results_csv_strips_padded_headers(results_csv: Path) -> None:
    columns = read_results_csv(results_csv)
    assert "metrics/mAP50(B)" in columns
    assert columns["epoch"][0] == 1.0
    assert len(columns["epoch"]) == 20


def test_results_csv_requires_epoch(tmp_path: Path) -> None:
    bad = tmp_path / "results.csv"
    bad.write_text("step,loss\n1,0.5\n")
    with pytest.raises(PlotError, match="epoch"):
        read_results_csv(bad)


def test_results_csv_rejects_non_numeric(tmp_path: Path) -> None:
    bad = tmp_path / "results.csv"
    bad.write_text("epoch,loss\n1,oops\n")
    with pytest.raises(PlotError, match="non-numeric"):
        read_results_csv(bad)


def test_detections_top_score_grouping(tmp_path: Path) -> None:
    path = _write_detections(
        tmp_path / "d.jsonl", {"a.jpg": [0.2, 0.9, 0.5], "b.jpg": [0.4]}
    )
    tops = top_scores_per_frame(read_detections_jsonl(path))
    assert sorted(tops) == [0.4, 0.9]


def test_detections_rejects_malformed_line(tmp_path: Path) -> None:
    bad = tmp_path / "d.jsonl"
    bad.write_text('{"image_id": "a.jpg", "score": 0.5}\n')
    with pytest.raises(PlotError, match="malformed detection line"):
        read_detections_jsonl(bad)


def test_detections_rejects_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "d.jsonl"
    empty.write_text("")
    with pytest.raises(PlotError, match="no data lines"):
        read_detections_jsonl(empty)
