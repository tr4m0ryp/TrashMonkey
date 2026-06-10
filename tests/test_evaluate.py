"""Tests for the T6 three-tier evaluation (task 011). ultralytics is fully mocked."""

import dataclasses
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
import yaml

import yolo_waste_sorter.models.evaluation as evaluation
from yolo_waste_sorter.models.evaluation import (
    CurveSet,
    EvalError,
    conf_at_precision,
    evaluate,
    extract_curves,
    load_manifest_index,
    load_report,
    materialize_severity,
)
from yolo_waste_sorter.utils.config import Config, load_config

CLASSES = ("plastic", "paper", "cardboard", "metal", "glass", "organic")
VAL_KEYS = ("srcA/v0.jpg", "srcA/v1.jpg", "srcB/v2.jpg")
TEST_KEYS = ("realwaste/t0.jpg", "realwaste/t1.jpg")
GROUPS = {  # v0 and v1 are two photos of the same physical object
    "srcA/v0.jpg": "srcA/v0.jpg",
    "srcA/v1.jpg": "srcA/v0.jpg",
    "srcB/v2.jpg": "srcB/v2.jpg",
}
SEVERITIES = (1, 2)


@pytest.fixture(scope="module")
def cfg() -> Config:
    base = load_config()
    return dataclasses.replace(
        base, eval=dataclasses.replace(base.eval, test2_severities=SEVERITIES)
    )


@pytest.fixture()
def dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Tiny emitted dataset + split manifest + fake best.pt."""
    root = tmp_path / "dataset"
    rng = np.random.default_rng(42)
    for split, keys in (("val", VAL_KEYS), ("test", TEST_KEYS)):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for key in keys:
            flat = key.replace("/", "__")
            img = rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
            assert cv2.imwrite(str(root / "images" / split / flat), img)
            label = root / "labels" / split / (Path(flat).stem + ".txt")
            label.write_text("0 0.5 0.5 0.2 0.2\n")
    data_yaml = root / "dataset.yaml"
    spec = {
        "path": str(root.resolve()),
        "train": "images/val",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(CLASSES)),
    }
    data_yaml.write_text(yaml.safe_dump(spec, sort_keys=False))
    manifest = tmp_path / "split_manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "stage": "split",
                "seed": 42,
                "val_fraction": 0.15,
                "leave_out_source": "realwaste",
                "counts": {},
                "groups": GROUPS,
                "assignments": {
                    **{key: "val" for key in VAL_KEYS},
                    **{key: "test" for key in TEST_KEYS},
                },
            }
        )
    )
    best_pt = tmp_path / "weights" / "best.pt"
    best_pt.parent.mkdir()
    best_pt.write_bytes(b"fake checkpoint")
    return data_yaml, manifest, best_pt


# --- mock ultralytics -------------------------------------------------------------


def _fake_metrics(
    map50: float = 0.97,
    per_class: dict[str, dict[str, float]] | None = None,
    classes: tuple[str, ...] = CLASSES,
) -> SimpleNamespace:
    defaults = {n: {"precision": 0.94, "recall": 0.95, "map50": 0.96, "map": 0.80} for n in classes}
    for name, override in (per_class or {}).items():
        defaults[name].update(override)
    points = np.linspace(0.0, 1.0, 21)
    # precision rises through 0.95 at confidence 0.5; recall/f1 fall with conf
    p_curve = np.tile(np.clip(0.6 + 0.75 * points, 0.0, 1.0), (len(classes), 1))
    r_curve = np.tile(np.clip(1.0 - points, 0.0, 1.0), (len(classes), 1))
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + 1e-9)
    box = SimpleNamespace(
        ap_class_index=list(range(len(classes))),
        p=[defaults[n]["precision"] for n in classes],
        r=[defaults[n]["recall"] for n in classes],
        ap50=[defaults[n]["map50"] for n in classes],
        ap=[defaults[n]["map"] for n in classes],
        map50=map50,
        map=0.80,
    )
    return SimpleNamespace(
        box=box,
        names=dict(enumerate(classes)),
        results_dict={"metrics/mAP50(B)": map50, "fitness": 0.85},
        curves_results=[
            [points, p_curve[:, ::-1], "Recall", "Precision"],  # PR decoy
            [points, f1_curve, "Confidence", "F1"],
            [points, p_curve, "Confidence", "Precision"],
            [points, r_curve, "Confidence", "Recall"],
        ],
    )


def _install_fake_ultralytics(
    monkeypatch: pytest.MonkeyPatch, val_results: list[SimpleNamespace] | None = None
) -> dict[str, Any]:
    calls: dict[str, Any] = {"val": [], "predict": []}
    queue = list(val_results) if val_results is not None else []

    class FakeYOLO:
        def __init__(self, model: str) -> None:
            calls["model"] = model

        def val(self, **kwargs: Any) -> SimpleNamespace:
            calls["val"].append(kwargs)
            return queue.pop(0) if queue else _fake_metrics()

        def predict(self, source: list[str], **kwargs: Any) -> Any:
            calls["predict"].append({"source": list(source), **kwargs})
            return iter(
                SimpleNamespace(path=p, boxes=SimpleNamespace(cls=[2.0], conf=[0.61]))
                for p in source
            )

    module = types.ModuleType("ultralytics")
    module.__version__ = "8.3.253"  # type: ignore[attr-defined]
    module.YOLO = FakeYOLO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    return calls


def _evaluate(
    cfg: Config, dataset: tuple[Path, Path, Path], tmp_path: Path
) -> Any:
    data_yaml, manifest, best_pt = dataset
    return evaluate(cfg, best_pt, data_yaml, manifest, out_dir=tmp_path / "evaluation")


# --- tier orchestration (T6) ------------------------------------------------------


def test_tier_order_and_sweep_conf(
    cfg: Config, dataset: tuple[Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_fake_ultralytics(monkeypatch)
    report = _evaluate(cfg, dataset, tmp_path)
    assert calls["model"] == str(dataset[2])
    assert [(Path(c["data"]).parent.name, c["split"]) for c in calls["val"]] == [
        ("dataset", "val"),
        ("dataset", "test"),
        ("severity_1", "test"),
        ("severity_2", "test"),
    ]
    assert all(c["conf"] == 0.001 for c in calls["val"])
    assert all(c["imgsz"] == cfg.model.imgsz for c in calls["val"])
    assert report.val.severity == 0 and report.test1.severity == 0
    assert [t.severity for t in report.test2] == [1, 2]
    assert [p.severity for p in report.severity_curve] == [1, 2]
    assert report.severity_curve[0].map50 == report.test2[0].map50


def test_no_selection_hooks_on_test_tiers() -> None:
    forbidden = ("tune", "select", "pick", "sweep_best", "choose")
    assert not [n for n in evaluation.__all__ if any(f in n.lower() for f in forbidden)]


def test_tier_report_skips_absent_classes(
    cfg: Config, dataset: tuple[Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    five = CLASSES[:5]  # organic missing from TEST-1 ground truth
    results = [_fake_metrics(), _fake_metrics(classes=five)]
    results += [_fake_metrics() for _ in SEVERITIES]
    _install_fake_ultralytics(monkeypatch, results)
    report = _evaluate(cfg, dataset, tmp_path)
    assert set(report.test1.per_class) == set(five)
    assert set(report.val.per_class) == set(CLASSES)


# --- degraded copies (TEST-2) -----------------------------------------------------


def test_materialize_severity_layout_and_determinism(
    dataset: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    data_yaml, _, _ = dataset
    yaml_a = materialize_severity(data_yaml, ("test", "val"), 3, 42, tmp_path / "a")
    yaml_b = materialize_severity(data_yaml, ("test", "val"), 3, 42, tmp_path / "b")
    spec = yaml.safe_load(yaml_a.read_text())
    assert spec["test"] == "images/test" and spec["val"] == "images/val"
    assert "train" in spec and spec["names"] == {i: n for i, n in enumerate(CLASSES)}
    for split, count in (("test", len(TEST_KEYS)), ("val", len(VAL_KEYS))):
        images_a = sorted((yaml_a.parent / "images" / split).glob("*.png"))
        labels_a = sorted((yaml_a.parent / "labels" / split).glob("*.txt"))
        assert len(images_a) == count and len(labels_a) == count
        for image in images_a:  # deterministic: byte-identical across runs
            twin = yaml_b.parent / "images" / split / image.name
            assert image.read_bytes() == twin.read_bytes()
        for label in labels_a:  # labels unchanged
            original = Path(yaml.safe_load(data_yaml.read_text())["path"])
            assert label.read_text() == (original / "labels" / split / label.name).read_text()


def test_materialize_severity_differs_across_severities(
    dataset: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    data_yaml, _, _ = dataset
    yaml_1 = materialize_severity(data_yaml, ("test",), 1, 42, tmp_path / "w")
    yaml_5 = materialize_severity(data_yaml, ("test",), 5, 42, tmp_path / "w")
    name = TEST_KEYS[0].replace("/", "__").replace(".jpg", ".png")
    assert (yaml_1.parent / "images" / "test" / name).read_bytes() != (
        yaml_5.parent / "images" / "test" / name
    ).read_bytes()


# --- report schema + round-trip ----------------------------------------------------


def test_report_yaml_round_trip(
    cfg: Config, dataset: tuple[Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_ultralytics(monkeypatch)
    report = _evaluate(cfg, dataset, tmp_path)
    path = tmp_path / "evaluation" / "eval_report.yaml"
    assert path.is_file()
    assert load_report(path) == report
    raw = yaml.safe_load(path.read_text())
    assert raw["stage"] == "evaluate"
    assert raw["seed"] == 42 and raw["conf_sweep"] == 0.001
    entry = raw["val"]["per_class"]["plastic"]
    assert set(entry) == {"precision", "recall", "map50", "map50_95", "conf_at_p95"}
    for tier in (report.val, report.test1, *report.test2):
        curves = np.load(tier.curves_path)
        assert set(curves.files) >= {"classes", "confidence", "precision", "recall", "f1"}
        assert curves["precision"].shape == (len(curves["classes"]), len(curves["confidence"]))


# --- escalation (T7, reused rule) ---------------------------------------------------


def test_escalation_edge_recall_089_fails(
    cfg: Config, dataset: tuple[Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = [_fake_metrics(per_class={"glass": {"recall": 0.89}})]
    results += [_fake_metrics() for _ in range(1 + len(SEVERITIES))]
    _install_fake_ultralytics(monkeypatch, results)
    report = _evaluate(cfg, dataset, tmp_path)
    assert report.escalation["passed"] is False
    assert report.escalation["per_class"]["glass"]["passed"] is False
    assert report.escalation["per_class"]["plastic"]["passed"] is True


def test_escalation_uses_val_not_test1(
    cfg: Config, dataset: tuple[Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = [_fake_metrics(), _fake_metrics(map50=0.50)]  # TEST-1 collapses
    results += [_fake_metrics() for _ in SEVERITIES]
    _install_fake_ultralytics(monkeypatch, results)
    report = _evaluate(cfg, dataset, tmp_path)
    assert report.escalation["passed"] is True  # escalation reads VAL only


# --- detections dump (T9 input) ------------------------------------------------------


def test_detections_jsonl_schema_and_object_ids(
    cfg: Config, dataset: tuple[Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_fake_ultralytics(monkeypatch)
    report = _evaluate(cfg, dataset, tmp_path)
    assert all(c["conf"] == cfg.thresholds.conf_floor for c in calls["predict"])
    lines = [json.loads(line) for line in Path(report.detections_path).read_text().splitlines()]
    assert len(lines) == len(VAL_KEYS) * (1 + len(SEVERITIES))  # one det per frame
    assert all(
        set(line) == {"image_id", "object_id", "class_id", "score", "severity"}
        for line in lines
    )
    assert {line["severity"] for line in lines} == {0, *SEVERITIES}
    assert {line["image_id"] for line in lines} == set(VAL_KEYS)  # val split only
    for line in lines:
        assert line["object_id"] == GROUPS[line["image_id"]]
        assert line["class_id"] == 2 and line["score"] == pytest.approx(0.61)


def test_manifest_stem_collision_fails_fast(tmp_path: Path) -> None:
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "assignments": {"a/x.jpg": "val", "a/x.png": "val"},
                "groups": {"a/x.jpg": "a/x.jpg", "a/x.png": "a/x.png"},
            }
        )
    )
    with pytest.raises(EvalError, match="stem collision"):
        load_manifest_index(manifest)


# --- curve extraction (P >= 0.95 confidence, verified 8.3.x surface) -----------------


def test_extract_curves_locates_confidence_precision_entry() -> None:
    curves = extract_curves(_fake_metrics())
    assert curves.classes == CLASSES
    assert curves.precision.shape == (len(CLASSES), 21)
    # the Recall/Precision PR decoy entry must NOT be picked up
    assert curves.precision[0, 0] == pytest.approx(0.6)


def test_conf_at_precision_sustained_region() -> None:
    confidence = np.linspace(0.0, 1.0, 11)
    spike_then_dip = [0.5, 0.5, 0.5, 0.5, 0.5, 0.96, 0.94, 0.97, 0.98, 0.99, 1.0]
    always_above = [0.96] * 11
    never_sustained = [0.5] * 10 + [0.94]
    curves = CurveSet(
        classes=("plastic", "paper", "metal"),
        confidence=confidence,
        precision=np.array([spike_then_dip, always_above, never_sustained]),
        recall=np.zeros((3, 11)),
        f1=np.zeros((3, 11)),
    )
    thresholds = conf_at_precision(curves, 0.95)
    assert thresholds["plastic"] == pytest.approx(0.7)  # dip at 0.6 disqualifies 0.5
    assert thresholds["paper"] == pytest.approx(0.0)
    assert thresholds["metal"] is None


def test_report_stores_conf_at_p95(
    cfg: Config, dataset: tuple[Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_ultralytics(monkeypatch)
    report = _evaluate(cfg, dataset, tmp_path)
    # fake curve: p = 0.6 + 0.75*conf, first sustained >= 0.95 at conf 0.5 (21-point grid)
    for name in CLASSES:
        assert report.val.per_class[name].conf_at_p95 == pytest.approx(0.5)
