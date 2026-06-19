"""Tests for the training entrypoint (task 009). ultralytics is fully mocked."""

import dataclasses
import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from trashmonkey.models.training import (
    SMOKE_OVERRIDES,
    build_smoke_dataset,
    build_train_kwargs,
    check_escalation,
    extract_metrics,
    train,
    validate_train_config,
)
from trashmonkey.models.training.escalation import (
    CLASS_MAP50_FLOOR,
    CLASS_RECALL_FLOOR,
    OVERALL_MAP50_FLOOR,
)
from trashmonkey.utils.config import Config, EscalationConfig, load_config

CLASSES = ("plastic", "paper", "cardboard", "metal", "glass", "organic")


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config()


@pytest.fixture(autouse=True)
def _no_ambient_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SMOKE_TEST", raising=False)


# --- mock ultralytics ----------------------------------------------------------


def _fake_results(
    map50: float = 0.97,
    per_class: dict[str, dict[str, float]] | None = None,
) -> SimpleNamespace:
    defaults = {name: {"map50": 0.96, "recall": 0.95} for name in CLASSES}
    if per_class:
        for name, override in per_class.items():
            defaults[name].update(override)
    names = dict(enumerate(CLASSES))
    box = SimpleNamespace(
        ap_class_index=list(range(len(CLASSES))),
        p=[0.94] * len(CLASSES),
        r=[defaults[name]["recall"] for name in CLASSES],
        ap50=[defaults[name]["map50"] for name in CLASSES],
        ap=[0.80] * len(CLASSES),
        map50=map50,
    )
    results_dict = {"metrics/mAP50(B)": map50, "fitness": 0.85}
    return SimpleNamespace(box=box, names=names, results_dict=results_dict)


def _install_fake_ultralytics(
    monkeypatch: pytest.MonkeyPatch,
    version: str = "8.3.253",
    results: SimpleNamespace | None = None,
) -> dict[str, Any]:
    """Inject a fake ultralytics module; returns a dict capturing the calls."""
    calls: dict[str, Any] = {}
    final_results = results if results is not None else _fake_results()

    class FakeYOLO:
        def __init__(self, model: str) -> None:
            calls["model"] = model
            self.trainer: SimpleNamespace | None = None

        def train(self, **kwargs: Any) -> SimpleNamespace:
            calls["train_kwargs"] = kwargs
            run_dir = Path(kwargs["project"]) / kwargs["name"]
            (run_dir / "weights").mkdir(parents=True, exist_ok=True)
            best = run_dir / "weights" / "best.pt"
            best.write_bytes(b"fake checkpoint")
            self.trainer = SimpleNamespace(
                best=str(best),
                save_dir=str(run_dir),
                metrics={"metrics/mAP50(B)": 0.97, "fitness": 0.85},
            )
            return final_results

    module = types.ModuleType("ultralytics")
    module.__version__ = version  # type: ignore[attr-defined]
    module.YOLO = FakeYOLO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    return calls


def _run(
    cfg: Config,
    tmp_path: Path,
    data_yaml: Path | None = None,
    smoke: bool = False,
    resume: Path | None = None,
    now: datetime | None = None,
) -> Any:
    if data_yaml is None and not smoke:
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("names: {0: plastic}\n")
    return train(
        cfg,
        data_yaml,
        smoke=smoke,
        runs_jsonl=tmp_path / "runs.jsonl",
        project=tmp_path / "runs",
        resume=resume,
        now=now or datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc),
    )


def _replace_train(cfg: Config, **changes: Any) -> Config:
    return dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, **changes))


# --- forbidden-arg guards (F1, F4, batch) ---------------------------------------


def test_optimizer_auto_rejected(cfg: Config) -> None:
    with pytest.raises(ValueError, match="optimizer='auto' is forbidden"):
        validate_train_config(_replace_train(cfg, optimizer="auto"))


def test_cache_ram_rejected(cfg: Config) -> None:
    with pytest.raises(ValueError, match="cache='ram'"):
        validate_train_config(_replace_train(cfg, cache="ram"))


def test_cache_true_rejected(cfg: Config) -> None:
    with pytest.raises(ValueError, match="cache='ram'"):
        validate_train_config(_replace_train(cfg, cache=True))


@pytest.mark.parametrize("batch", [-1, 0])
def test_non_positive_batch_rejected(cfg: Config, batch: int) -> None:
    with pytest.raises(ValueError, match="positive int"):
        validate_train_config(_replace_train(cfg, batch=batch))


def test_train_guards_fire_before_ultralytics_import(cfg: Config, tmp_path: Path) -> None:
    # No mock installed: a forbidden config must fail before any import.
    with pytest.raises(ValueError, match="optimizer"):
        _run(_replace_train(cfg, optimizer="auto"), tmp_path)


def test_old_ultralytics_rejected(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_ultralytics(monkeypatch, version="8.3.0")
    with pytest.raises(RuntimeError, match=r"8\.3\.226"):
        _run(cfg, tmp_path)


# --- config -> kwargs translation (T7 table snapshot) ---------------------------


def test_train_kwargs_snapshot(cfg: Config) -> None:
    assert build_train_kwargs(cfg, Path("data/processed/data.yaml")) == {
        "data": "data/processed/data.yaml",
        "epochs": 100,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "cos_lr": False,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "batch": 16,
        "imgsz": 640,
        "patience": 30,
        "close_mosaic": 10,
        "cache": "disk",
        "amp": True,
        "deterministic": True,
        "seed": 42,
        "workers": 8,
        "freeze": None,
        "cls_pw": 0.0,
        "degrees": 180.0,
        "flipud": 0.5,
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.5,
        "translate": 0.1,
        "scale": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "copy_paste": 0.0,
    }


def test_train_passes_kwargs_and_stack(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_fake_ultralytics(monkeypatch)
    result = _run(cfg, tmp_path)
    assert calls["model"] == "yolo11n.pt"
    kwargs = calls["train_kwargs"]
    expected = build_train_kwargs(cfg, Path(kwargs["data"]))
    assert {k: kwargs[k] for k in expected} == expected
    assert [type(t).__name__ for t in kwargs["augmentations"]] == [
        "ImageCompression",
        "ISONoise",
        "GaussNoise",
        "MotionBlur",
        "Defocus",
        "PlanckianJitter",
        "Downscale",
        "RandomBrightnessContrast",
    ]
    assert result.best_pt.is_file()
    assert result.run_dir.is_dir()


# --- runs.jsonl ------------------------------------------------------------------

_REQUIRED_RECORD_KEYS = {
    "timestamp",
    "git_commit",
    "config",
    "train_kwargs",
    "dataset_yaml",
    "dataset_yaml_sha256",
    "ultralytics_version",
    "device",
    "metrics",
    "escalation",
    "wall_clock_seconds",
    "run_dir",
    "best_pt",
    "smoke",
    "resumed_from",
}


def test_runs_jsonl_appends_one_valid_line(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_ultralytics(monkeypatch)
    now = datetime(2026, 6, 10, 9, 30, 0, tzinfo=timezone.utc)
    _run(cfg, tmp_path, now=now)
    lines = (tmp_path / "runs.jsonl").read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert _REQUIRED_RECORD_KEYS <= set(record)
    assert record["timestamp"] == now.isoformat()
    assert record["ultralytics_version"] == "8.3.253"
    assert record["config"]["seed"] == 42
    assert record["config"]["train"]["optimizer"] == "AdamW"
    assert len(record["dataset_yaml_sha256"]) == 64
    assert record["escalation"]["passed"] is True
    assert record["smoke"] is False
    assert record["resumed_from"] is None
    assert isinstance(record["wall_clock_seconds"], float)
    # the stack is serialized as reprs, not objects
    assert all(isinstance(t, str) for t in record["train_kwargs"]["augmentations"])

    _run(cfg, tmp_path, now=now)  # second run appends, never rewrites
    assert len((tmp_path / "runs.jsonl").read_text().splitlines()) == 2


# --- escalation rule (config-driven floors) --------------------------------------

# Floors used by the gate tests; these mirror the configs/config.yaml defaults
# and the module-level fallbacks (asserted equal below).
FLOORS = EscalationConfig(overall_map50=0.80, class_map50=0.70, class_recall=0.70)


def test_module_fallback_floors_match_config_defaults(cfg: Config) -> None:
    # The module constants are the ultimate fallback and must track the config.
    assert OVERALL_MAP50_FLOOR == cfg.eval.escalation.overall_map50
    assert CLASS_MAP50_FLOOR == cfg.eval.escalation.class_map50
    assert CLASS_RECALL_FLOOR == cfg.eval.escalation.class_recall
    assert (OVERALL_MAP50_FLOOR, CLASS_MAP50_FLOOR, CLASS_RECALL_FLOOR) != (0.95, 0.90, 0.90)


def test_escalation_passes_when_all_floors_met() -> None:
    block = check_escalation(extract_metrics(_fake_results()), CLASSES, FLOORS)
    assert block["passed"] is True
    assert set(block["per_class"]) == set(CLASSES)
    assert block["thresholds"] == {
        "overall_map50": 0.80,
        "class_map50": 0.70,
        "class_recall": 0.70,
    }


def test_escalation_floors_come_from_config_not_constants() -> None:
    # 0.85 recall passes the default 0.70 floor but fails a strict 0.90 floor.
    results = _fake_results(per_class={"glass": {"recall": 0.85}})
    lenient = check_escalation(extract_metrics(results), CLASSES, FLOORS)
    strict = check_escalation(
        extract_metrics(results),
        CLASSES,
        EscalationConfig(overall_map50=0.80, class_map50=0.70, class_recall=0.90),
    )
    assert lenient["per_class"]["glass"]["passed"] is True
    assert strict["per_class"]["glass"]["passed"] is False


def test_escalation_fails_on_single_class_recall_below_floor() -> None:
    results = _fake_results(per_class={"glass": {"recall": 0.69}})
    block = check_escalation(extract_metrics(results), CLASSES, FLOORS)
    assert block["passed"] is False
    assert block["per_class"]["glass"]["passed"] is False
    assert block["per_class"]["plastic"]["passed"] is True


def test_escalation_fails_on_single_class_map50_below_floor() -> None:
    results = _fake_results(per_class={"paper": {"map50": 0.69}})
    block = check_escalation(extract_metrics(results), CLASSES, FLOORS)
    assert block["passed"] is False
    assert block["per_class"]["paper"]["passed"] is False


def test_escalation_fails_on_low_overall_map50() -> None:
    block = check_escalation(extract_metrics(_fake_results(map50=0.799)), CLASSES, FLOORS)
    assert block["passed"] is False
    assert all(entry["passed"] for entry in block["per_class"].values())


def test_escalation_uses_module_fallback_when_floors_none() -> None:
    # No floors arg -> module fallbacks (== config defaults) apply.
    block = check_escalation(extract_metrics(_fake_results(map50=0.799)), CLASSES)
    assert block["passed"] is False
    assert block["thresholds"]["overall_map50"] == OVERALL_MAP50_FLOOR


def test_escalation_fails_on_missing_class() -> None:
    metrics = extract_metrics(_fake_results())
    del metrics["per_class"]["organic"]
    block = check_escalation(metrics, CLASSES, FLOORS)
    assert block["passed"] is False
    assert block["per_class"]["organic"] == {"map50": None, "recall": None, "passed": False}


# --- resume -----------------------------------------------------------------------


def _make_last_pt(tmp_path: Path) -> Path:
    last_pt = tmp_path / "prior" / "weights" / "last.pt"
    last_pt.parent.mkdir(parents=True)
    last_pt.write_bytes(b"fake interrupted checkpoint")
    return last_pt


def test_resume_loads_checkpoint_and_sets_flag(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_fake_ultralytics(monkeypatch)
    last_pt = _make_last_pt(tmp_path)
    _run(cfg, tmp_path, resume=last_pt)
    assert calls["model"] == str(last_pt)  # model loads from last.pt, not the base
    assert calls["train_kwargs"]["resume"] == str(last_pt)
    record = json.loads((tmp_path / "runs.jsonl").read_text().splitlines()[0])
    assert record["resumed_from"] == str(last_pt)


def test_resume_missing_checkpoint_rejected(cfg: Config, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="resume checkpoint missing"):
        _run(cfg, tmp_path, resume=tmp_path / "absent" / "last.pt")


def test_resume_rejected_in_smoke_mode(cfg: Config, tmp_path: Path) -> None:
    last_pt = _make_last_pt(tmp_path)
    with pytest.raises(ValueError, match="incompatible with smoke mode"):
        _run(cfg, tmp_path, smoke=True, resume=last_pt)


# --- smoke mode -------------------------------------------------------------------


def test_smoke_dataset_synthesis(tmp_path: Path) -> None:
    data_yaml = build_smoke_dataset(CLASSES, tmp_path / "dataset")
    spec = yaml.safe_load(data_yaml.read_text())
    assert spec["names"] == dict(enumerate(CLASSES))
    image_dir = Path(spec["path"]) / spec["train"]
    label_dir = Path(spec["path"]) / "labels" / "train"
    images = sorted(image_dir.glob("*.jpg"))
    labels = sorted(label_dir.glob("*.txt"))
    assert len(images) == 12 and len(labels) == 12
    assert [i.stem for i in images] == [label.stem for label in labels]
    seen_classes = set()
    for label in labels:
        fields = label.read_text().split()
        assert len(fields) == 5
        class_id = int(fields[0])
        seen_classes.add(class_id)
        assert 0 <= class_id < len(CLASSES)
        coords = [float(v) for v in fields[1:]]
        assert all(0.0 < value < 1.0 for value in coords)
    assert seen_classes == set(range(len(CLASSES)))


def test_smoke_mode_overrides_and_synthesizes(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_fake_ultralytics(monkeypatch)
    result = _run(cfg, tmp_path, data_yaml=None, smoke=True)
    kwargs = calls["train_kwargs"]
    assert {k: kwargs[k] for k in SMOKE_OVERRIDES} == SMOKE_OVERRIDES
    synthesized = Path(kwargs["data"])
    assert synthesized.name == "data.yaml" and synthesized.is_file()
    assert len(list(synthesized.parent.glob("images/train/*.jpg"))) == 12
    assert result.best_pt.is_file()
    record = json.loads((tmp_path / "runs.jsonl").read_text().splitlines()[0])
    assert record["smoke"] is True


def test_smoke_via_env_var(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_fake_ultralytics(monkeypatch)
    monkeypatch.setenv("SMOKE_TEST", "1")
    _run(cfg, tmp_path, data_yaml=None, smoke=False)
    assert calls["train_kwargs"]["epochs"] == 1


def test_data_yaml_required_outside_smoke(cfg: Config, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="data_yaml is required"):
        train(cfg, None, smoke=False, runs_jsonl=tmp_path / "runs.jsonl")
