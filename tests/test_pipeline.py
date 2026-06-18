"""Pipeline CLI tests: ordering, resume, --force, QA gate, and a real end-to-end run.

NO network, NO model downloads: fixture sources go through the local fetcher
and the autobox chain runs with injected fake backends.
"""

from __future__ import annotations

import dataclasses
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from trashmonkey.data.autobox import Detection
from trashmonkey.data.pipeline import (
    PipelineContext,
    PipelineHalt,
    Stage,
    StageError,
    build_context,
    build_stages,
    main,
    run_pipeline,
)
from trashmonkey.data.pipeline.stages.qa import _draw_sample
from trashmonkey.data.qa import ImageQA, QAReport

REPO_ROOT = Path(__file__).resolve().parents[1]


# --- fixture: two local cls sources + a tmp config/datasets pair ----------------


def png_bytes(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def make_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, blob in members.items():
            zf.writestr(name, blob)


def source_entry(
    name: str,
    archive: Path,
    mapping: dict[str, str],
    drops: list[str],
    box_order: list[str] | None = None,
    role: str | None = None,
) -> dict[str, object]:
    entry: dict[str, object] = {
        "name": name,
        "fetcher": {"kind": "local", "ref": str(archive), "sha256": None},
        "license": "MIT",
        "attribution": f"synthetic fixture source {name}",
        "annotation_type": "cls",
        "background": "clean",
        "mapping": mapping,
        "drops": drops,
    }
    if box_order is not None:
        entry["box_order"] = box_order
    if role is not None:
        entry["role"] = role
    return entry


def write_fixture(tmp_path: Path) -> Path:
    """alpha: plastic x2, paper, cardboard, metal + one DROP; beta: glass, organic.

    beta is the leave-out (TEST-1) source. Every config class gets a directory
    (scan_remapped requires all six) and the DROP image feeds the wilderness
    pool, with the smallest image count that still exercises every stage.
    """
    archives = tmp_path / "archives"
    archives.mkdir()
    make_zip(
        archives / "alpha.zip",
        {
            "bottle/a1.png": png_bytes(1),
            "bottle/a2.png": png_bytes(2),
            "sheet/a3.png": png_bytes(3),
            "box/a4.png": png_bytes(4),
            "can/a5.png": png_bytes(5),
            "junk/a6.png": png_bytes(6),
        },
    )
    make_zip(archives / "beta.zip", {"jar/b1.png": png_bytes(7), "peel/b2.png": png_bytes(8)})
    datasets = {
        "sources": [
            source_entry(
                "alpha",
                archives / "alpha.zip",
                {
                    "bottle": "plastic",
                    "sheet": "paper",
                    "box": "cardboard",
                    "can": "metal",
                    "junk": "DROP",
                },
                ["junk"],
            ),
            source_entry("beta", archives / "beta.zip", {"jar": "glass", "peel": "organic"}, []),
        ]
    }
    raw_cfg = yaml.safe_load((REPO_ROOT / "configs" / "config.yaml").read_text())
    raw_cfg["paths"] = {
        key: str(tmp_path / "data" / key) for key in ("raw", "interim", "processed", "external")
    } | {"models": str(tmp_path / "models"), "reports": str(tmp_path / "reports")}
    raw_cfg["eval"]["leave_out_source"] = "beta"
    raw_cfg["eval"]["val_fraction"] = 0.5
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text(yaml.safe_dump(raw_cfg))
    (cfg_dir / "datasets.yaml").write_text(yaml.safe_dump(datasets))
    return cfg_dir / "config.yaml"


def write_tiers_fixture(tmp_path: Path) -> Path:
    """write_fixture + a `gamma` test_only source and an alpha clean holdout.

    gamma (role=test_only, plastic) must land wholesale in `wild_test` and never
    in train/val/balance; alpha is the clean_holdout source so a `clean_test`
    tier is carved. beta stays the leave-out (TEST-1) source.
    """
    archives = tmp_path / "archives"
    archives.mkdir()
    make_zip(
        archives / "alpha.zip",
        {
            "bottle/a1.png": png_bytes(1),
            "bottle/a2.png": png_bytes(2),
            "bottle/a3.png": png_bytes(11),
            "bottle/a4.png": png_bytes(12),
            "sheet/a5.png": png_bytes(3),
            "box/a6.png": png_bytes(4),
            "can/a7.png": png_bytes(5),
            "junk/a8.png": png_bytes(6),
        },
    )
    make_zip(archives / "beta.zip", {"jar/b1.png": png_bytes(7), "peel/b2.png": png_bytes(8)})
    make_zip(
        archives / "gamma.zip",
        {"bottle/g1.png": png_bytes(9), "bottle/g2.png": png_bytes(10)},
    )
    datasets = {
        "sources": [
            source_entry(
                "alpha",
                archives / "alpha.zip",
                {
                    "bottle": "plastic",
                    "sheet": "paper",
                    "box": "cardboard",
                    "can": "metal",
                    "junk": "DROP",
                },
                ["junk"],
            ),
            source_entry("beta", archives / "beta.zip", {"jar": "glass", "peel": "organic"}, []),
            source_entry(
                "gamma",
                archives / "gamma.zip",
                {"bottle": "plastic"},
                [],
                role="test_only",
            ),
        ]
    }
    raw_cfg = yaml.safe_load((REPO_ROOT / "configs" / "config.yaml").read_text())
    raw_cfg["paths"] = {
        key: str(tmp_path / "data" / key) for key in ("raw", "interim", "processed", "external")
    } | {"models": str(tmp_path / "models"), "reports": str(tmp_path / "reports")}
    raw_cfg["eval"]["leave_out_source"] = "beta"
    raw_cfg["eval"]["val_fraction"] = 0.5
    raw_cfg["eval"]["clean_holdout"] = {"fraction": 0.5, "sources": ["alpha"]}
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text(yaml.safe_dump(raw_cfg))
    (cfg_dir / "datasets.yaml").write_text(yaml.safe_dump(datasets))
    return cfg_dir / "config.yaml"


def fake_dino(image_path: Path) -> list[Detection]:
    with Image.open(image_path) as img:
        width, height = img.size
    box = (width * 0.25, height * 0.25, width * 0.75, height * 0.75)
    return [Detection(xyxy=box, confidence=0.9)]


def fail_mask(image_path: Path) -> object:
    raise AssertionError(f"birefnet fallback must not fire for {image_path}")


@pytest.fixture
def ctx(tmp_path: Path) -> PipelineContext:
    base = build_context(write_fixture(tmp_path))
    return dataclasses.replace(base, dino_predict=fake_dino, birefnet_mask=fail_mask)


# --- fake-stage runner behaviour ------------------------------------------------


def fake_stage(
    name: str,
    log: list[str],
    completed: set[str],
    *,
    fail: bool = False,
    halt_unless_ack: bool = False,
) -> Stage:
    def run(ctx: PipelineContext) -> str:
        if halt_unless_ack and not ctx.ack_review:
            raise PipelineHalt(f"{name}: review needed")
        if fail:
            raise RuntimeError("boom")
        log.append(name)
        completed.add(name)
        return "ok"

    return Stage(name=name, run=run, is_complete=lambda ctx: name in completed, hint=f"fix {name}")


def test_stages_run_in_declared_order(ctx: PipelineContext) -> None:
    log: list[str] = []
    done: set[str] = set()
    stages = [fake_stage(n, log, done) for n in ("a", "b", "c")]
    actions = run_pipeline(stages, ctx)
    assert log == ["a", "b", "c"]
    assert actions == {"a": "ran", "b": "ran", "c": "ran"}


def test_on_stage_fires_once_per_stage_in_order(ctx: PipelineContext) -> None:
    log: list[str] = []
    done: set[str] = set()
    stages = [fake_stage(n, log, done) for n in ("a", "b", "c")]
    seen: list[tuple[str, int, int]] = []
    run_pipeline(stages, ctx, on_stage=lambda name, i, total: seen.append((name, i, total)))
    assert seen == [("a", 1, 3), ("b", 2, 3), ("c", 3, 3)]


def _run_through_autobox(ctx: PipelineContext) -> None:
    for stage_name in ("download", "remap", "autobox"):
        next(s for s in build_stages() if s.name == stage_name).run(ctx)


def test_autobox_resumes_from_checkpoints_when_manifest_absent(ctx: PipelineContext) -> None:
    calls = {"n": 0}

    def counting_dino(path: Path) -> list[Detection]:
        calls["n"] += 1
        return fake_dino(path)

    ctx2 = dataclasses.replace(ctx, dino_predict=counting_dino)
    _run_through_autobox(ctx2)
    first = calls["n"]
    assert first == 8  # 7 cls images + 1 wilderness, boxed once

    # Simulate a crash AFTER the groups checkpointed but before the next run:
    # drop the stage manifest, keep the per-group checkpoints.
    ctx2.manifest_path("autobox").unlink()
    next(s for s in build_stages() if s.name == "autobox").run(ctx2)
    assert calls["n"] == first  # every group loaded from checkpoint, none re-boxed
    assert ctx2.manifest_path("autobox").is_file()  # manifest rebuilt


def test_autobox_force_rerun_reboxes_when_manifest_present(ctx: PipelineContext) -> None:
    calls = {"n": 0}

    def counting_dino(path: Path) -> list[Detection]:
        calls["n"] += 1
        return fake_dino(path)

    ctx2 = dataclasses.replace(ctx, dino_predict=counting_dino)
    _run_through_autobox(ctx2)
    first = calls["n"]
    # Manifest present -> a direct re-run is a forced re-box (checkpoints dropped).
    next(s for s in build_stages() if s.name == "autobox").run(ctx2)
    assert calls["n"] == 2 * first


def test_autobox_reports_cumulative_per_image_progress(ctx: PipelineContext) -> None:
    # Fixture cls images: alpha {plastic x2, paper, cardboard, metal} + beta
    # {glass, organic} = 7, plus one DROP image in the wilderness pool = 8.
    seen: list[tuple[str, int, int]] = []
    ctx2 = dataclasses.replace(
        ctx, progress=lambda label, cur, total: seen.append((label, cur, total))
    )
    for stage_name in ("download", "remap", "autobox"):
        next(s for s in build_stages() if s.name == stage_name).run(ctx2)
    assert seen, "autobox reported no progress"
    assert {label for label, _, _ in seen} == {"autobox"}
    assert {total for _, _, total in seen} == {8}
    assert [cur for _, cur, _ in seen] == list(range(1, 9))  # cumulative 1..8


def write_box_order_fixture(tmp_path: Path) -> Path:
    """Like write_fixture but beta carries box_order=[birefnet,dino,centerbox]."""
    archives = tmp_path / "archives"
    archives.mkdir()
    make_zip(
        archives / "alpha.zip",
        {
            "bottle/a1.png": png_bytes(1),
            "bottle/a2.png": png_bytes(2),
            "sheet/a3.png": png_bytes(3),
            "box/a4.png": png_bytes(4),
            "can/a5.png": png_bytes(5),
            "junk/a6.png": png_bytes(6),
        },
    )
    make_zip(archives / "beta.zip", {"jar/b1.png": png_bytes(7), "peel/b2.png": png_bytes(8)})
    datasets = {
        "sources": [
            source_entry(
                "alpha",
                archives / "alpha.zip",
                {
                    "bottle": "plastic",
                    "sheet": "paper",
                    "box": "cardboard",
                    "can": "metal",
                    "junk": "DROP",
                },
                ["junk"],
            ),
            source_entry(
                "beta",
                archives / "beta.zip",
                {"jar": "glass", "peel": "organic"},
                [],
                box_order=["birefnet", "dino", "centerbox"],
            ),
        ]
    }
    raw_cfg = yaml.safe_load((REPO_ROOT / "configs" / "config.yaml").read_text())
    raw_cfg["paths"] = {
        key: str(tmp_path / "data" / key) for key in ("raw", "interim", "processed", "external")
    } | {"models": str(tmp_path / "models"), "reports": str(tmp_path / "reports")}
    raw_cfg["eval"]["leave_out_source"] = "beta"
    raw_cfg["eval"]["val_fraction"] = 0.5
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text(yaml.safe_dump(raw_cfg))
    (cfg_dir / "datasets.yaml").write_text(yaml.safe_dump(datasets))
    return cfg_dir / "config.yaml"


def test_autobox_stage_threads_per_source_box_order(tmp_path: Path) -> None:
    """beta (box_order=[birefnet,dino,centerbox]) segments first -- its images
    are boxed by birefnet and DINO is never queried for them; alpha (no
    box_order) keeps dino-first. The stage reads each source's order from the
    registry and passes it to the chain."""
    base = build_context(write_box_order_fixture(tmp_path))

    dino_seen: list[str] = []

    def recording_dino(image_path: Path) -> list[Detection]:
        dino_seen.append(image_path.name)
        return fake_dino(image_path)

    def good_mask(image_path: Path) -> np.ndarray:
        # A 50%-of-image centered blob: passes the area gates -> birefnet wins.
        with Image.open(image_path) as img:
            width, height = img.size
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = 255
        return mask

    ctx2 = dataclasses.replace(base, dino_predict=recording_dino, birefnet_mask=good_mask)
    for stage_name in ("download", "remap", "autobox"):
        next(s for s in build_stages() if s.name == stage_name).run(ctx2)

    autobox_root = ctx2.interim_root / "autobox"
    methods: dict[str, str] = {}
    for class_name in ctx2.cfg.classes:
        prov = autobox_root / class_name / "provenance.jsonl"
        if not prov.is_file():
            continue
        for line in prov.read_text().splitlines():
            rec = json.loads(line)
            methods[Path(rec["image"]).name] = rec["method"]

    # beta segments first: birefnet method, DINO never queried for its images.
    # Remap names images <source>__<stem>; the class-name prefix is added later.
    assert methods["beta__b1.png"] == "birefnet"
    assert methods["beta__b2.png"] == "birefnet"
    assert not any("beta__" in name for name in dino_seen)
    # alpha keeps dino-first (fake_dino always returns a confident box).
    assert methods["alpha__a1.png"] == "dino"
    assert any("alpha__" in name for name in dino_seen)


def test_resume_skips_completed_stages(ctx: PipelineContext) -> None:
    log: list[str] = []
    done = {"a", "b"}  # stages 1-2 already complete
    stages = [fake_stage(n, log, done) for n in ("a", "b", "c")]
    actions = run_pipeline(stages, ctx)
    assert log == ["c"]
    assert actions == {"a": "skipped", "b": "skipped", "c": "ran"}
    log.clear()
    assert run_pipeline(stages, ctx) == {n: "skipped" for n in ("a", "b", "c")}
    assert log == []


def test_force_reruns_completed_stages(ctx: PipelineContext) -> None:
    log: list[str] = []
    done = {"a", "b", "c"}
    stages = [fake_stage(n, log, done) for n in ("a", "b", "c")]
    assert run_pipeline(stages, ctx, force=True) == {n: "ran" for n in ("a", "b", "c")}
    assert log == ["a", "b", "c"]


def test_force_from_named_stage(ctx: PipelineContext) -> None:
    log: list[str] = []
    done = {"a", "b", "c"}
    stages = [fake_stage(n, log, done) for n in ("a", "b", "c")]
    actions = run_pipeline(stages, ctx, start="b", force=True)
    assert log == ["b", "c"]
    assert actions == {"a": "skipped", "b": "ran", "c": "ran"}


def test_start_stage_requires_earlier_complete(ctx: PipelineContext) -> None:
    stages = [fake_stage(n, [], set()) for n in ("a", "b")]
    with pytest.raises(StageError, match="stage 'a' is not complete"):
        run_pipeline(stages, ctx, start="b")
    with pytest.raises(StageError, match="unknown stage 'z'"):
        run_pipeline(stages, ctx, start="z")


def test_failure_names_stage_and_hint(ctx: PipelineContext) -> None:
    log: list[str] = []
    stages = [
        fake_stage("a", log, set()),
        fake_stage("b", log, set(), fail=True),
        fake_stage("c", log, set()),
    ]
    with pytest.raises(StageError, match=r"stage 'b' failed: boom -- hint: fix b"):
        run_pipeline(stages, ctx)
    assert log == ["a"]  # stopped on first failure


def test_qa_halt_and_ack_review(ctx: PipelineContext) -> None:
    log: list[str] = []
    done: set[str] = set()
    stages = [
        fake_stage("a", log, done),
        fake_stage("qa", log, done, halt_unless_ack=True),
        fake_stage("c", log, done),
    ]
    with pytest.raises(PipelineHalt, match="review needed"):
        run_pipeline(stages, ctx)
    assert log == ["a"] and "qa" not in done
    acked = dataclasses.replace(ctx, ack_review=True)
    assert run_pipeline(stages, acked) == {"a": "skipped", "qa": "ran", "c": "ran"}
    assert log == ["a", "qa", "c"]


# --- QA sampling: paper + plastic oversampled ------------------------------------


def report_of(group: str, n: int) -> QAReport:
    images = {
        f"{group}{i}": ImageQA(
            image=f"{group}{i}.png",
            stem=f"{group}{i}",
            source="alpha",
            method="dino",
            confidence=0.9,
            n_boxes=1,
            class_id=0,
        )
        for i in range(n)
    }
    return QAReport(labels_dir=f"/tmp/{group}", images=images)


def test_review_sample_oversamples_paper_and_plastic() -> None:
    reports = {"paper": report_of("paper", 20), "metal": report_of("metal", 20)}
    drawn = _draw_sample(reports, seed=42)
    by_group = {g: sum(1 for grp, _ in drawn if grp == g) for g in reports}
    assert by_group == {"paper": 4, "metal": 2}  # ceil(0.20*20) vs ceil(0.10*20)


# --- integration: real stage wrappers over the tmpdir fixture --------------------


def test_full_pipeline_halts_at_qa_then_resumes_to_dataset(ctx: PipelineContext) -> None:
    stages = build_stages()
    with pytest.raises(PipelineHalt, match="--ack-review"):
        run_pipeline(stages, ctx)
    interim = ctx.interim_root
    assert (interim / "pipeline" / "autobox.yaml").is_file()
    assert (interim / "review" / "sample.csv").is_file()
    assert not (interim / "pipeline" / "qa.yaml").exists()  # gate blocked completion
    # wilderness pool boxed with placeholder class 0
    wilderness_label = interim / "wilderness" / "alpha__a6.txt"
    assert wilderness_label.read_text().startswith("0 ")
    assert (interim / "autobox" / "wilderness" / "provenance.jsonl").is_file()

    acked = dataclasses.replace(ctx, ack_review=True)
    actions = run_pipeline(stages, acked)
    assert actions == {
        "download": "skipped",
        "remap": "skipped",
        "autobox": "skipped",
        "qa": "ran",
        "dedup": "ran",
        "balance": "ran",
        "split": "ran",
    }

    root = ctx.processed_root / ctx.cfg.experiment.name
    spec = yaml.safe_load((root / "dataset.yaml").read_text())
    assert spec["names"] == dict(enumerate(ctx.cfg.classes))
    for split in ("train", "val", "test"):
        images = sorted((root / "images" / split).iterdir())
        assert images, f"empty split: {split}"
        for image in images:
            assert (root / "labels" / split / f"{image.stem}.txt").is_file()
    # TEST-1: the leave-out source is exactly the test split
    test_names = {p.name for p in (root / "images" / "test").iterdir()}
    assert test_names == {"glass__beta__b1.png", "organic__beta__b2.png"}
    trainval = [
        p.name for s in ("train", "val") for p in (root / "images" / s).iterdir()
    ]
    assert all("beta__" not in name for name in trainval)

    # idempotent rerun: everything now skips
    assert run_pipeline(stages, acked) == {name: "skipped" for name in actions}


def test_full_pipeline_emits_clean_and_wild_test_tiers(tmp_path: Path) -> None:
    base = build_context(write_tiers_fixture(tmp_path))
    ctx = dataclasses.replace(
        base, ack_review=True, dino_predict=fake_dino, birefnet_mask=fail_mask
    )
    run_pipeline(build_stages(), ctx)

    root = ctx.processed_root / ctx.cfg.experiment.name
    spec = yaml.safe_load((root / "dataset.yaml").read_text())
    for split in ("train", "val", "test", "clean_test", "wild_test"):
        assert spec[split] == f"images/{split}", f"missing split {split} in dataset.yaml"
        assert (root / "images" / split).is_dir()

    # wild_test holds EXACTLY the gamma (role=test_only) images.
    wild = {p.name for p in (root / "images" / "wild_test").iterdir()}
    assert wild == {"plastic__gamma__g1.png", "plastic__gamma__g2.png"}
    # clean_test is carved from alpha only.
    clean = {p.name for p in (root / "images" / "clean_test").iterdir()}
    assert clean and all("alpha__" in n for n in clean)
    # train/val exclude both the leave-out (beta) and the test_only (gamma) source.
    trainval = [
        p.name for s in ("train", "val") for p in (root / "images" / s).iterdir()
    ]
    assert all("beta__" not in n and "gamma__" not in n for n in trainval)
    # labels travel with images for every emitted split.
    for split in ("train", "val", "test", "clean_test", "wild_test"):
        for image in (root / "images" / split).iterdir():
            assert (root / "labels" / split / f"{image.stem}.txt").is_file()

    # the balance manifest records the label-filter drop summary.
    balance = yaml.safe_load(ctx.manifest_path("balance").read_text())
    assert "label_filter" in balance
    assert "dropped" in balance["label_filter"]
    split_manifest = yaml.safe_load(ctx.manifest_path("split").read_text())
    assert split_manifest["test_only_sources"] == ["gamma"]
    assert split_manifest["clean_holdout_sources"] == ["alpha"]


def test_pipeline_is_seeded_and_deterministic(tmp_path: Path) -> None:
    manifests = []
    for run_dir in (tmp_path / "one", tmp_path / "two"):
        run_dir.mkdir()
        base = build_context(write_fixture(run_dir))
        run_ctx = dataclasses.replace(
            base, ack_review=True, dino_predict=fake_dino, birefnet_mask=fail_mask
        )
        run_pipeline(build_stages(), run_ctx)
        manifest = yaml.safe_load(run_ctx.manifest_path("split").read_text())
        manifests.append((manifest["assignments"], manifest["counts"]))
    assert manifests[0] == manifests[1]


# --- CLI + Makefile wiring --------------------------------------------------------


def test_cli_reports_config_error(tmp_path: Path) -> None:
    assert main(["run", "--config", str(tmp_path / "missing.yaml")]) == 1


def test_make_repro_invokes_pipeline_cli() -> None:
    recipe = (REPO_ROOT / "Makefile").read_text().split("repro:")[1]
    assert "-m trashmonkey.data.pipeline run" in recipe
    assert "TODO" not in recipe
