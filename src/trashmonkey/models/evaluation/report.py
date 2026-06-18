"""YAML-serializable evaluation report dataclasses (T6 three-tier eval).

``EvalReport`` round-trips through ``write_yaml``/``load_report`` so the
threshold tuner (012) and the plot stage (014) consume the exact numbers the
evaluation run produced. Severity 0 means the clean (undegraded) images.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CLEAN_SEVERITY = 0


class EvalError(Exception):
    """Evaluation inputs or intermediate artifacts are malformed."""


@dataclass(frozen=True)
class ClassEval:
    """Per-class metrics at one tier, plus the T9 threshold anchor."""

    precision: float
    recall: float
    map50: float
    map50_95: float
    # Smallest confidence from which precision stays >= 0.95 (None: never).
    conf_at_p95: float | None


@dataclass(frozen=True)
class TierReport:
    """One evaluation tier: VAL, TEST-1, or one TEST-2 severity level."""

    tier: str  # "val" | "test1" | "test2_s<severity>"
    split: str  # dataset split the tier ran on ("val" | "test")
    severity: int  # 0 = clean, 1..5 = ESP32 degradation level
    map50: float
    map50_95: float
    overall: dict[str, float]  # ultralytics results_dict, verbatim
    per_class: dict[str, ClassEval]  # only classes present in the tier's GT
    curves_path: str  # .npz with confidence/precision/recall/f1 arrays


@dataclass(frozen=True)
class SeverityPoint:
    """One point of the TEST-2 severity curve."""

    severity: int
    map50: float
    map50_95: float


@dataclass(frozen=True)
class EvalReport:
    """Full report: per-tier metrics, severity curve, escalation, dumps.

    ``clean`` is the deployment-matched CLEAN tier (the ``clean_test`` split)
    and ``wild`` is the in-the-wild stress tier (the ``wild_test`` split); both
    are None for datasets emitted without those splits. ``headline`` surfaces
    the CLEAN-tier mAP50 + per-class recall (with mAP50-95 as a secondary
    field) -- it falls back to VAL when no CLEAN tier was run.
    """

    seed: int
    best_pt: str
    data_yaml: str
    classes: tuple[str, ...]
    conf_sweep: float  # conf passed to val (full-curve sweep)
    val: TierReport
    test1: TierReport
    test2: tuple[TierReport, ...]
    severity_curve: tuple[SeverityPoint, ...]
    escalation: dict[str, Any]  # training/escalation.py check_escalation block
    detections_path: str  # JSONL the threshold tuner (012) replays
    clean: TierReport | None = None  # deployment-matched CLEAN tier (clean_test)
    wild: TierReport | None = None  # in-the-wild stress tier (wild_test)
    headline: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["classes"] = list(self.classes)
        data["test2"] = [dataclasses.asdict(tier) for tier in self.test2]
        data["severity_curve"] = [dataclasses.asdict(p) for p in self.severity_curve]
        return {"stage": "evaluate", **data}

    def write_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def _class_eval_from(data: dict[str, Any], where: str) -> ClassEval:
    try:
        return ClassEval(**data)
    except TypeError as exc:
        raise EvalError(f"{where}: malformed per-class block: {exc}") from exc


def _tier_from(data: dict[str, Any], where: str) -> TierReport:
    fields = {f.name for f in dataclasses.fields(TierReport)}
    missing = sorted(fields - set(data))
    if missing:
        raise EvalError(f"{where}: missing tier key(s): {', '.join(missing)}")
    per_class = {
        str(name): _class_eval_from(block, f"{where}.per_class.{name}")
        for name, block in data["per_class"].items()
    }
    return TierReport(
        tier=str(data["tier"]),
        split=str(data["split"]),
        severity=int(data["severity"]),
        map50=float(data["map50"]),
        map50_95=float(data["map50_95"]),
        overall={str(k): float(v) for k, v in data["overall"].items()},
        per_class=per_class,
        curves_path=str(data["curves_path"]),
    )


def _optional_tier_from(data: dict[str, Any] | None, where: str) -> TierReport | None:
    """A tier that may be absent (CLEAN/WILD tiers on old 3-split datasets)."""
    return None if data is None else _tier_from(data, where)


def report_from_dict(data: dict[str, Any]) -> EvalReport:
    """Rebuild an ``EvalReport`` from its ``to_dict`` form (fail fast)."""
    if data.get("stage") != "evaluate":
        raise EvalError(f"not an evaluation report: stage={data.get('stage')!r}")
    # clean/wild/headline are optional: old reports predate the CLEAN tier.
    optional = {"clean", "wild", "headline"}
    required = [f.name for f in dataclasses.fields(EvalReport) if f.name not in optional]
    missing = sorted(set(required) - set(data))
    if missing:
        raise EvalError(f"report missing key(s): {', '.join(missing)}")
    return EvalReport(
        seed=int(data["seed"]),
        best_pt=str(data["best_pt"]),
        data_yaml=str(data["data_yaml"]),
        classes=tuple(str(c) for c in data["classes"]),
        conf_sweep=float(data["conf_sweep"]),
        val=_tier_from(data["val"], "val"),
        test1=_tier_from(data["test1"], "test1"),
        test2=tuple(
            _tier_from(tier, f"test2[{i}]") for i, tier in enumerate(data["test2"])
        ),
        severity_curve=tuple(
            SeverityPoint(
                severity=int(p["severity"]),
                map50=float(p["map50"]),
                map50_95=float(p["map50_95"]),
            )
            for p in data["severity_curve"]
        ),
        escalation=dict(data["escalation"]),
        detections_path=str(data["detections_path"]),
        clean=_optional_tier_from(data.get("clean"), "clean"),
        wild=_optional_tier_from(data.get("wild"), "wild"),
        headline=dict(data.get("headline") or {}),
    )


def load_report(path: Path) -> EvalReport:
    """Load and validate a report written by ``EvalReport.write_yaml``."""
    if not path.is_file():
        raise EvalError(f"report file not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise EvalError(f"{path}: top level must be a mapping")
    return report_from_dict(raw)
