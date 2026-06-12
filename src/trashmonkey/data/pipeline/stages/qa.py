"""QA stage: automated checks, stratified review sample, review queue, human gate.

run_checks covers 100% of the autobox-generated labels per group (one group
per class with cls-source images, plus the wilderness pool, so the placeholder
class 0 never pollutes another class's outlier statistics). A ~10% stratified
review sample is drawn per group -- paper and plastic at double the fraction
per the T3 review plan -- and written to data/interim/review/sample.csv; every
flagged image lands in the per-group review queue.

The T3 acceptance metrics (review fail rate, localization fail rate, median
IoU) come from HUMAN review of that sample, so they are unknowable in-pipeline:
the gate treats pending metrics as not passed and HALTS with the review-queue
path. After reviewing, rerun with --ack-review to continue.
"""

from __future__ import annotations

import csv
from pathlib import Path

from trashmonkey.data.autobox import PROVENANCE_FILENAME
from trashmonkey.data.pipeline.context import PipelineContext, PipelineHalt
from trashmonkey.data.pipeline.runner import Stage
from trashmonkey.data.qa import ImageQA, QAReport, emit_review_queue, run_checks
from trashmonkey.data.qa import stratified_sample as qa_stratified_sample

REVIEW_FRACTION = 0.10
OVERSAMPLE_FRACTION = 0.20  # paper + plastic are oversampled per the T3 review plan
OVERSAMPLE_CLASSES = frozenset({"paper", "plastic"})
SAMPLE_CSV_NAME = "sample.csv"
SAMPLE_CSV_FIELDS = ("image", "group", "class_id", "source", "method", "confidence", "flags")


def _groups(ctx: PipelineContext) -> list[str]:
    if not ctx.autobox_root.is_dir():
        return []
    return sorted(
        entry.name
        for entry in ctx.autobox_root.iterdir()
        if entry.is_dir() and (entry / PROVENANCE_FILENAME).is_file()
    )


def _draw_sample(reports: dict[str, QAReport], seed: int) -> list[tuple[str, ImageQA]]:
    drawn: list[tuple[str, ImageQA]] = []
    for group in sorted(reports):
        items = sorted(reports[group].images.values(), key=lambda record: record.stem)
        frac = OVERSAMPLE_FRACTION if group in OVERSAMPLE_CLASSES else REVIEW_FRACTION
        for item in qa_stratified_sample(items, frac, seed=seed):
            drawn.append((group, item))
    return drawn


def _write_sample(path: Path, drawn: list[tuple[str, ImageQA]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SAMPLE_CSV_FIELDS)
        for group, item in drawn:
            writer.writerow(
                [
                    item.image,
                    group,
                    item.class_id,
                    item.source,
                    item.method,
                    item.confidence,
                    "|".join(item.flags),
                ]
            )


def _gate(reports: dict[str, QAReport], ctx: PipelineContext) -> str:
    """'passed' | 'acked' | 'no-autobox-labels'; raises PipelineHalt otherwise."""
    if not reports:
        return "no-autobox-labels"
    pending = any(
        report.review_fail_rate is None
        or report.loc_fail_rate is None
        or report.median_iou is None
        for report in reports.values()
    )
    failed = not pending and not all(r.acceptance_pass for r in reports.values())
    if not pending and not failed:
        return "passed"
    if ctx.ack_review:
        return "acked"
    reason = (
        "acceptance metrics pending human review"
        if pending
        else "acceptance bars failed (see data/interim/qa/*.yaml)"
    )
    raise PipelineHalt(
        f"QA gate: {reason}. Review the queue at {ctx.review_root} "
        f"(sample: {ctx.review_root / SAMPLE_CSV_NAME}), then rerun with --ack-review."
    )


def _qa_run(ctx: PipelineContext) -> str:
    reports: dict[str, QAReport] = {}
    for group in _groups(ctx):
        labels_dir = ctx.autobox_root / group
        report = run_checks(labels_dir, labels_dir / PROVENANCE_FILENAME)
        report.to_yaml(ctx.qa_root / f"{group}.yaml")
        reports[group] = report

    drawn = _draw_sample(reports, ctx.cfg.seed)
    ctx.review_root.mkdir(parents=True, exist_ok=True)
    _write_sample(ctx.review_root / SAMPLE_CSV_NAME, drawn)
    for group, report in reports.items():
        emit_review_queue(report, ctx.review_root / group)

    gate = _gate(reports, ctx)
    total = sum(report.total_images for report in reports.values())
    flagged = sum(len(report.flagged) for report in reports.values())
    ctx.write_manifest(
        "qa",
        {
            "groups": {
                group: {"total": report.total_images, "flagged": len(report.flagged)}
                for group, report in sorted(reports.items())
            },
            "sample_size": len(drawn),
            "sample_csv": str(ctx.review_root / SAMPLE_CSV_NAME),
            "review_dir": str(ctx.review_root),
            "gate": gate,
        },
    )
    return (
        f"{total} labels checked, {flagged} flagged, review sample of {len(drawn)} "
        f"-> {ctx.review_root / SAMPLE_CSV_NAME} (gate: {gate})"
    )


def qa_stage() -> Stage:
    return Stage(
        name="qa",
        run=_qa_run,
        is_complete=lambda ctx: ctx.manifest_path("qa").is_file(),
        hint=(
            "inspect data/interim/qa/*.yaml and the review queue under "
            "data/interim/review/; rerun with --ack-review after human review"
        ),
    )
