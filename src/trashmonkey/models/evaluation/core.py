"""evaluate(): the T6 three-tier evaluation of a trained checkpoint.

Tier order: VAL (selection ceiling) -> TEST-1 (held-out source split) ->
TEST-2 (degraded TEST-1 copies per severity, reported as a curve). Every val
runs at ``conf=0.001`` for the full curve sweep; the T7 escalation check
reuses ``models/training/escalation.py``. This module deliberately exposes NO
tuning or selection hooks on TEST-1/TEST-2 -- both are report-only tiers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trashmonkey.models.evaluation.curves import (
    conf_at_precision,
    extract_curves,
    save_curves,
)
from trashmonkey.models.evaluation.degraded import materialize_severity, split_images
from trashmonkey.models.evaluation.detections import (
    dump_detections,
    load_manifest_index,
)
from trashmonkey.models.evaluation.report import (
    CLEAN_SEVERITY,
    ClassEval,
    EvalError,
    EvalReport,
    SeverityPoint,
    TierReport,
)
from trashmonkey.models.training.escalation import check_escalation, extract_metrics
from trashmonkey.utils.config import Config
from trashmonkey.utils.seed import set_seed

SWEEP_CONF = 0.001  # full-curve sweep; never a deployment threshold
REPORT_FILENAME = "eval_report.yaml"
DETECTIONS_FILENAME = "detections.jsonl"


def _free_gpu() -> None:
    """Release cached CUDA memory between passes (7 val() runs + the dumps leave
    the reserved pool fragmented, OOMing the final dump on a 40GB card)."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _run_val(
    model: Any, data_yaml: Path, split: str, cfg: Config, out_dir: Path, name: str
) -> Any:
    return model.val(
        data=str(data_yaml),
        split=split,
        conf=SWEEP_CONF,
        imgsz=cfg.model.imgsz,
        plots=False,
        verbose=False,
        project=str(out_dir / "runs"),
        name=name,
    )


def _tier_report(
    results: Any, tier: str, split: str, severity: int, out_dir: Path
) -> TierReport:
    metrics = extract_metrics(results)
    curves = extract_curves(results)
    curves_path = save_curves(curves, out_dir / "curves" / f"{tier}.npz")
    thresholds = conf_at_precision(curves)
    if set(thresholds) != set(metrics["per_class"]):
        raise EvalError(
            f"tier {tier}: curve classes {sorted(thresholds)} do not match "
            f"per-class metrics {sorted(metrics['per_class'])}"
        )
    per_class = {
        name: ClassEval(
            precision=block["precision"],
            recall=block["recall"],
            map50=block["map50"],
            map50_95=block["map50_95"],
            conf_at_p95=thresholds[name],
        )
        for name, block in metrics["per_class"].items()
    }
    return TierReport(
        tier=tier,
        split=split,
        severity=severity,
        map50=metrics["map50"],
        map50_95=float(results.box.map),
        overall=metrics["overall"],
        per_class=per_class,
        curves_path=str(curves_path),
    )


def evaluate(
    cfg: Config,
    best_pt: Path,
    data_yaml: Path,
    split_manifest_path: Path,
    *,
    out_dir: Path | None = None,
    work_dir: Path | None = None,
) -> EvalReport:
    """Run the three tiers, the escalation check, and the detections dump.

    Args:
        cfg: Typed experiment config (eval.test2_severities, thresholds, seed).
        best_pt: Trained checkpoint to evaluate.
        data_yaml: The emitted dataset yaml (val + test splits).
        split_manifest_path: ``SplitResult.write_manifest`` output; provides
            the instance-group ids stamped on every detections line.
        out_dir: Report/curves/detections destination (default:
            ``<run_dir>/evaluation`` next to the checkpoint).
        work_dir: Degraded-copy working directory (default: ``out_dir/degraded``).
    """
    if not best_pt.is_file():
        raise EvalError(f"checkpoint not found: {best_pt}")
    if not cfg.eval.test2_severities:
        raise EvalError("cfg.eval.test2_severities is empty -- TEST-2 needs >= 1 level")
    index = load_manifest_index(split_manifest_path)
    out_dir = best_pt.parent.parent / "evaluation" if out_dir is None else out_dir
    work_dir = out_dir / "degraded" if work_dir is None else work_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)
    import ultralytics  # lazy: the package must import without it installed

    model = ultralytics.YOLO(str(best_pt))

    val_results = _run_val(model, data_yaml, "val", cfg, out_dir, "val")
    val_tier = _tier_report(val_results, "val", "val", CLEAN_SEVERITY, out_dir)
    _free_gpu()
    test1_results = _run_val(model, data_yaml, "test", cfg, out_dir, "test1")
    test1_tier = _tier_report(test1_results, "test1", "test", CLEAN_SEVERITY, out_dir)
    _free_gpu()

    test2_tiers: list[TierReport] = []
    severity_yamls: dict[int, Path] = {}
    for severity in cfg.eval.test2_severities:
        # The val split is degraded in the same pass for the tuner dump (T9).
        severity_yaml = materialize_severity(
            data_yaml, ("test", "val"), severity, cfg.seed, work_dir
        )
        severity_yamls[severity] = severity_yaml
        tier = f"test2_s{severity}"
        results = _run_val(model, severity_yaml, "test", cfg, out_dir, tier)
        test2_tiers.append(_tier_report(results, tier, "test", severity, out_dir))
        _free_gpu()
    severity_curve = tuple(
        SeverityPoint(severity=t.severity, map50=t.map50, map50_95=t.map50_95)
        for t in test2_tiers
    )

    escalation = check_escalation(extract_metrics(val_results), cfg.classes)

    _free_gpu()  # reclaim the val/test fragmentation before the dump's predict
    detections_path = out_dir / DETECTIONS_FILENAME
    conf = cfg.thresholds.conf_floor
    imgsz = cfg.model.imgsz
    with open(detections_path, "w", encoding="utf-8") as out:
        dump_detections(
            model, split_images(data_yaml, "val"), CLEAN_SEVERITY, index, out, conf=conf, imgsz=imgsz
        )
        for severity in cfg.eval.test2_severities:
            images = split_images(severity_yamls[severity], "val")
            dump_detections(model, images, severity, index, out, conf=conf, imgsz=imgsz)

    report = EvalReport(
        seed=cfg.seed,
        best_pt=str(best_pt),
        data_yaml=str(data_yaml),
        classes=cfg.classes,
        conf_sweep=SWEEP_CONF,
        val=val_tier,
        test1=test1_tier,
        test2=tuple(test2_tiers),
        severity_curve=severity_curve,
        escalation=escalation,
        detections_path=str(detections_path),
    )
    report.write_yaml(out_dir / REPORT_FILENAME)
    return report
