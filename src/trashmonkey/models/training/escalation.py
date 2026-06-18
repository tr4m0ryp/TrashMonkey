"""Metrics extraction and the T7 escalation rule.

``extract_metrics`` reads the duck-typed DetMetrics surface returned by
ultralytics ``model.train()`` (verified against v8.3.x ``utils/metrics.py``:
``results_dict``, ``names``, and per-class arrays on ``.box`` aligned with
``box.ap_class_index``). ``check_escalation`` records the pass/fail numbers;
the decision to escalate to yolo11s stays human.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trashmonkey.utils.config import EscalationConfig

# Ultimate fallback floors when no EscalationConfig is supplied; these mirror
# the configs/config.yaml ``eval.escalation`` defaults (the source of truth).
OVERALL_MAP50_FLOOR = 0.80
CLASS_MAP50_FLOOR = 0.70
CLASS_RECALL_FLOOR = 0.70


def extract_metrics(results: Any) -> dict[str, Any]:
    """Parse overall and per-class metrics from a DetMetrics-like object."""
    overall = {str(key): float(value) for key, value in results.results_dict.items()}
    names = {int(class_id): str(name) for class_id, name in results.names.items()}
    box = results.box
    per_class: dict[str, dict[str, float]] = {}
    for i, class_id in enumerate(box.ap_class_index):
        per_class[names[int(class_id)]] = {
            "precision": float(box.p[i]),
            "recall": float(box.r[i]),
            "map50": float(box.ap50[i]),
            "map50_95": float(box.ap[i]),
        }
    return {"overall": overall, "map50": float(box.map50), "per_class": per_class}


def check_escalation(
    metrics: dict[str, Any],
    classes: tuple[str, ...],
    floors: EscalationConfig | None = None,
) -> dict[str, Any]:
    """Apply the escalation rule with config-driven per-metric floors.

    Escalate (``passed`` is False) when overall mAP50 drops below
    ``floors.overall_map50`` or any per-class mAP50/recall drops below
    ``floors.class_map50``/``floors.class_recall``. When ``floors`` is None the
    module-level fallbacks (which mirror the config defaults) are used.

    A class absent from the validation results cannot demonstrate its floors
    and is recorded as failed with null numbers -- no silent pass.
    """
    overall_floor = OVERALL_MAP50_FLOOR if floors is None else floors.overall_map50
    class_map50_floor = CLASS_MAP50_FLOOR if floors is None else floors.class_map50
    class_recall_floor = CLASS_RECALL_FLOOR if floors is None else floors.class_recall
    passed = metrics["map50"] >= overall_floor
    per_class: dict[str, dict[str, Any]] = {}
    for name in classes:
        entry = metrics["per_class"].get(name)
        if entry is None:
            per_class[name] = {"map50": None, "recall": None, "passed": False}
            passed = False
            continue
        class_ok = entry["map50"] >= class_map50_floor and entry["recall"] >= class_recall_floor
        per_class[name] = {"map50": entry["map50"], "recall": entry["recall"], "passed": class_ok}
        passed = passed and class_ok
    return {
        "passed": passed,
        "overall_map50": metrics["map50"],
        "per_class": per_class,
        "thresholds": {
            "overall_map50": overall_floor,
            "class_map50": class_map50_floor,
            "class_recall": class_recall_floor,
        },
    }
