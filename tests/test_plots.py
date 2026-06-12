"""Figure rendering tests (task 014): fixture artifacts only, Agg backend, no display."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from trashmonkey.data.split import SplitResult
from trashmonkey.models.evaluation.curves import CurveSet, save_curves
from trashmonkey.models.evaluation.report import (
    ClassEval,
    EvalReport,
    SeverityPoint,
    TierReport,
)
from trashmonkey.visualization.plots import (
    PlotError,
    SweepRow,
    load_curves_npz,
    pareto_front,
    plot_dataset_composition,
    plot_per_class_curves,
    plot_severity_curve,
    plot_threshold_tradeoff,
    read_split_composition,
    read_sweep_csv,
)

matplotlib.use("Agg", force=True)

CLASSES = ("plastic", "paper", "cardboard", "metal", "glass", "organic")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
SWEEP_HEADER = "tau_frame,min_votes,high_water,wrong_bin_rate,rest_rate,chosen"


def _assert_png(path: Path, min_bytes: int = 5_000) -> None:
    assert path.is_file()
    data = path.read_bytes()
    assert data[:8] == PNG_MAGIC
    assert len(data) > min_bytes
    assert plt.get_fignums() == []  # every plot function closes its figure


# --- fixtures ---------------------------------------------------------------------


def _curve_set() -> CurveSet:
    conf = np.linspace(0.0, 1.0, 101)
    precision = np.stack(
        [np.clip(0.55 + 0.5 * conf + 0.02 * i, 0.0, 1.0) for i in range(len(CLASSES))]
    )
    recall = np.stack(
        [np.clip(1.0 - 0.85 * conf - 0.02 * i, 0.0, 1.0) for i in range(len(CLASSES))]
    )
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return CurveSet(
        classes=CLASSES, confidence=conf, precision=precision, recall=recall, f1=f1
    )


def _tier(tier: str, split: str, severity: int, curves_path: str, map50: float) -> TierReport:
    per_class = {
        name: ClassEval(
            precision=0.94,
            recall=0.92,
            map50=map50,
            map50_95=map50 - 0.17,
            conf_at_p95=None if name == "organic" else 0.3 + 0.05 * i,
        )
        for i, name in enumerate(CLASSES)
    }
    return TierReport(
        tier=tier,
        split=split,
        severity=severity,
        map50=map50,
        map50_95=map50 - 0.17,
        overall={"metrics/mAP50(B)": map50},
        per_class=per_class,
        curves_path=curves_path,
    )


@pytest.fixture()
def curves_npz(tmp_path: Path) -> Path:
    return save_curves(_curve_set(), tmp_path / "curves_val.npz")


@pytest.fixture()
def report(tmp_path: Path, curves_npz: Path) -> EvalReport:
    return EvalReport(
        seed=42,
        best_pt="weights/best.pt",
        data_yaml="dataset.yaml",
        classes=CLASSES,
        conf_sweep=0.001,
        val=_tier("val", "val", 0, str(curves_npz), 0.95),
        test1=_tier("test1", "test", 0, str(tmp_path / "curves_test1.npz"), 0.91),
        test2=tuple(
            _tier(f"test2_s{s}", "test", s, str(tmp_path / f"curves_t2s{s}.npz"), 0.91 - 0.1 * s)
            for s in (1, 2, 3)
        ),
        severity_curve=(
            SeverityPoint(severity=1, map50=0.81, map50_95=0.64),
            SeverityPoint(severity=2, map50=0.71, map50_95=0.54),
            SeverityPoint(severity=3, map50=0.61, map50_95=0.44),
            SeverityPoint(severity=5, map50=0.38, map50_95=0.24),
        ),
        escalation={"escalate": False},
        detections_path="detections.jsonl",
    )


@pytest.fixture()
def sweep_csv(tmp_path: Path) -> Path:
    rows = [SWEEP_HEADER]
    chosen_written = False
    for i, tau in enumerate((0.3, 0.4, 0.5, 0.6, 0.7)):
        for votes in (2, 3):
            wrong = round(0.05 - 0.006 * i - 0.004 * votes, 4)
            rest = round(0.02 + 0.02 * i + 0.015 * votes, 4)
            chosen = int(not chosen_written and wrong <= 0.02)
            chosen_written = chosen_written or bool(chosen)
            rows.append(f"{tau},{votes},0.9,{wrong},{rest},{chosen}")
    path = tmp_path / "sweep.csv"
    path.write_text("\n".join(rows) + "\n")
    return path


ASSIGNMENTS = {
    "cardboard/trashnet__0009.jpg": "train",
    "glass/trashnet__0007.jpg": "train",
    "metal/realwaste__0011.jpg": "test",
    "metal/taco__0006.jpg": "train",
    "organic/taco__0008.jpg": "train",
    "paper/taco__0005.jpg": "val",
    "paper/trashnet__0004.jpg": "train",
    "plastic/realwaste__0010.jpg": "test",
    "plastic/taco__0001.jpg": "train",
    "plastic/taco__0002.jpg": "train",
    "plastic/trashnet__0003.jpg": "val",
}


@pytest.fixture()
def split_manifest(tmp_path: Path) -> Path:
    counts: dict[str, dict[str, int]] = {}
    for key, split in ASSIGNMENTS.items():
        class_name = key.split("/", 1)[0]
        per_class = counts.setdefault(split, {})
        per_class[class_name] = per_class.get(class_name, 0) + 1
    result = SplitResult(
        assignments=dict(ASSIGNMENTS),
        group_ids={key: key for key in ASSIGNMENTS},
        counts=counts,
        seed=42,
        val_fraction=0.15,
        leave_out_source="realwaste",
    )
    path = tmp_path / "split_manifest.yaml"
    result.write_manifest(path)
    return path


# --- figure rendering -------------------------------------------------------------


def test_per_class_curves_from_report_path(
    tmp_path: Path, report: EvalReport, curves_npz: Path
) -> None:
    report_path = tmp_path / "eval_report.yaml"
    report.write_yaml(report_path)  # exercises the real writer + load_report
    out = tmp_path / "figs" / "per_class_curves.png"
    plot_per_class_curves(report_path, curves_npz, save_path=out)
    _assert_png(out)


def test_per_class_curves_unmatched_npz_falls_back_to_val(
    tmp_path: Path, report: EvalReport
) -> None:
    other = save_curves(_curve_set(), tmp_path / "unmatched_tier.npz")
    out = tmp_path / "fallback.png"
    plot_per_class_curves(report, other, save_path=out)
    _assert_png(out)


def test_severity_curve_starts_at_clean_test1(tmp_path: Path, report: EvalReport) -> None:
    out = tmp_path / "severity.png"
    plot_severity_curve(report, save_path=out)
    _assert_png(out)


def test_threshold_tradeoff(tmp_path: Path, sweep_csv: Path) -> None:
    out = tmp_path / "tradeoff.png"
    plot_threshold_tradeoff(sweep_csv, save_path=out)
    _assert_png(out)


def test_dataset_composition(tmp_path: Path, split_manifest: Path) -> None:
    out = tmp_path / "composition.png"
    plot_dataset_composition(split_manifest, save_path=out)
    _assert_png(out)


def test_render_without_save_path_closes_figure(report: EvalReport) -> None:
    plot_severity_curve(report, save_path=None)
    assert plt.get_fignums() == []


# --- parsing and contracts ---------------------------------------------------------


def test_sweep_csv_roundtrip_and_chosen(sweep_csv: Path) -> None:
    rows = read_sweep_csv(sweep_csv)
    assert len(rows) == 10
    assert sum(row.chosen for row in rows) == 1
    assert all(isinstance(row.min_votes, int) for row in rows)


def test_pareto_front_is_non_dominated() -> None:
    def cell(wrong: float, rest: float) -> SweepRow:
        return SweepRow(
            tau_frame=0.5, min_votes=2, high_water=0.9,
            wrong_bin_rate=wrong, rest_rate=rest, chosen=False,
        )

    rows = (cell(0.05, 0.01), cell(0.01, 0.10), cell(0.03, 0.05), cell(0.03, 0.20))
    front = pareto_front(rows)
    assert [(r.wrong_bin_rate, r.rest_rate) for r in front] == [
        (0.01, 0.10), (0.03, 0.05), (0.05, 0.01),
    ]


def test_sweep_csv_rejects_wrong_columns(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("tau,votes,high,wrong,rest,chosen\n0.5,2,0.9,0.01,0.1,1\n")
    with pytest.raises(PlotError, match="contract"):
        read_sweep_csv(bad)


def test_sweep_csv_rejects_non_binary_chosen(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text(f"{SWEEP_HEADER}\n0.5,2,0.9,0.01,0.1,2\n")
    with pytest.raises(PlotError, match="chosen"):
        read_sweep_csv(bad)


def test_curves_npz_rejects_missing_arrays(tmp_path: Path) -> None:
    path = tmp_path / "partial.npz"
    np.savez_compressed(path, classes=np.asarray(CLASSES), confidence=np.linspace(0, 1, 5))
    with pytest.raises(PlotError, match="missing array"):
        load_curves_npz(path)


def test_split_composition_counts(split_manifest: Path) -> None:
    counts = read_split_composition(split_manifest)
    assert counts["train"]["plastic"] == {"taco": 2}
    assert counts["val"]["plastic"] == {"trashnet": 1}
    assert counts["test"] == {"metal": {"realwaste": 1}, "plastic": {"realwaste": 1}}


def test_split_composition_rejects_wrong_stage(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text("stage: dedup\nassignments: {}\n")
    with pytest.raises(PlotError, match="split-stage"):
        read_split_composition(path)


def test_split_composition_rejects_malformed_key(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text("stage: split\nassignments:\n  no-source-separator.jpg: train\n")
    with pytest.raises(PlotError, match="assignment key"):
        read_split_composition(path)
