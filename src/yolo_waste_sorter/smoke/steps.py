"""Post-pipeline smoke steps: train, evaluate, wilderness dump, threshold sweep.

Each step returns a ``StepResult`` (one-line summary + key artifact path) for
the harness's PASS lines. The wilderness dump replays the evaluation module's
``dump_detections`` over the interim wilderness pool with a synthetic manifest
index (``wilderness/<stem>`` object ids -- disjoint from val keys by
construction), feeding the tuner's unknown-object probe.
"""

from __future__ import annotations

import dataclasses
import shutil
from dataclasses import dataclass
from pathlib import Path

from yolo_waste_sorter.data.remap import IMAGE_SUFFIXES
from yolo_waste_sorter.models.evaluation import EvalReport, ManifestIndex, dump_detections, evaluate
from yolo_waste_sorter.models.thresholding import TuneResult, truth_from_manifest, tune_thresholds
from yolo_waste_sorter.models.training import RunResult, train
from yolo_waste_sorter.smoke.workspace import REPO_ROOT
from yolo_waste_sorter.utils.config import Config

WILDERNESS_DETECTIONS = "wilderness_detections.jsonl"


@dataclass(frozen=True)
class StepResult:
    """One smoke step's PASS-line payload."""

    summary: str
    artifact: Path


def _cached_base(cfg: Config) -> Config:
    """Point model.base at models/<name> when a cached checkpoint exists."""
    cached = REPO_ROOT / "models" / cfg.model.base
    if "/" not in cfg.model.base and cached.is_file():
        return dataclasses.replace(
            cfg, model=dataclasses.replace(cfg.model, base=str(cached))
        )
    return cfg


def _cache_downloaded_base(cfg: Config) -> None:
    """Reuse: keep an ultralytics-downloaded bare-name checkpoint under models/."""
    base = cfg.model.base
    cached = REPO_ROOT / "models" / base
    downloaded = Path.cwd() / base
    if "/" not in base and downloaded.is_file() and not cached.is_file():
        cached.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, cached)


def run_training(cfg: Config, data_yaml: Path, workdir: Path) -> tuple[RunResult, StepResult]:
    """1-epoch CPU smoke train (SMOKE_OVERRIDES) against the produced dataset."""
    run_cfg = _cached_base(cfg)
    result = train(
        run_cfg,
        data_yaml,
        smoke=True,
        runs_jsonl=workdir / "experiments" / "runs.jsonl",
        project=workdir / "experiments" / "runs",
    )
    _cache_downloaded_base(run_cfg)
    summary = (
        f"1-epoch smoke train, mAP50={result.metrics['map50']:.3f}, "
        f"run log -> {workdir / 'experiments' / 'runs.jsonl'}"
    )
    return result, StepResult(summary=summary, artifact=result.best_pt)


def run_evaluation(
    cfg: Config, best_pt: Path, data_yaml: Path, split_manifest: Path, workdir: Path
) -> tuple[EvalReport, StepResult]:
    """VAL + TEST-1 + severity-1 TEST-2 tiers plus the detections dump."""
    out_dir = workdir / "evaluation"
    report = evaluate(cfg, best_pt, data_yaml, split_manifest, out_dir=out_dir)
    summary = (
        f"val mAP50={report.val.map50:.3f}, test1 mAP50={report.test1.map50:.3f}, "
        f"{len(report.test2)} TEST-2 tier(s), detections -> {report.detections_path}"
    )
    return report, StepResult(summary=summary, artifact=out_dir / "eval_report.yaml")


def _wilderness_index(images: list[Path]) -> ManifestIndex:
    """Synthetic per-image identity: object_id 'wilderness/<stem>' (no grouping)."""
    keys = {image.stem: f"wilderness/{image.stem}" for image in images}
    return ManifestIndex(
        key_by_stem=keys,
        group_by_key={key: key for key in keys.values()},
        split_by_key={key: "wilderness" for key in keys.values()},
    )


def dump_wilderness(cfg: Config, best_pt: Path, wilderness_root: Path, workdir: Path) -> StepResult:
    """Predict over the DROP-routed pool; JSONL probe for the tuner."""
    images = sorted(
        p for p in wilderness_root.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    ) if wilderness_root.is_dir() else []
    if not images:
        raise RuntimeError(
            f"no wilderness images under {wilderness_root} -- the fixture must "
            "contain DROP-class images so the unknown-object probe is exercised"
        )
    import ultralytics  # lazy: resolves to the injected fake under FAKE_MODEL=1

    model = ultralytics.YOLO(str(best_pt))
    out_path = workdir / "evaluation" / WILDERNESS_DETECTIONS
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        written = dump_detections(
            model, images, 0, _wilderness_index(images), out, conf=cfg.thresholds.conf_floor
        )
    return StepResult(
        summary=f"{len(images)} wilderness image(s), {written} detection line(s)",
        artifact=out_path,
    )


def run_thresholds(
    cfg: Config, report: EvalReport, split_manifest: Path, wilderness_jsonl: Path, workdir: Path
) -> tuple[TuneResult, StepResult]:
    """Reduced-grid T9 sweep over the dumped detections + wilderness probe."""
    truth = truth_from_manifest(split_manifest, cfg.classes)
    result = tune_thresholds(
        cfg,
        Path(report.detections_path),
        truth,
        workdir / "thresholds",
        wilderness_jsonl=wilderness_jsonl,
        report=report,
    )
    summary = (
        f"{len(result.cells)} sweep cell(s), chose min_votes={result.params.min_votes} "
        f"high_water={result.params.high_water} (wrong_bin={result.wrong_bin_rate:.4f}, "
        f"rest={result.rest_rate:.4f}, constraint_met={result.constraint_met}), "
        f"sweep -> {result.sweep_path}"
    )
    return result, StepResult(summary=summary, artifact=result.thresholds_path)
