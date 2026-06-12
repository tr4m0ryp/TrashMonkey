"""Remap-stage tests: routing, label rewrite, drops, collisions, raw immutability.

All fixtures live in tmpdirs; image "files" are fake bytes (the remap stage
never decodes pixels). No network, no GPU.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from trashmonkey.data.download import FetcherSpec, SourceSpec
from trashmonkey.data.remap import (
    ClassNamesError,
    RemapManifest,
    UnmappedLabelError,
    manifest_path,
    remap_source,
)

TARGETS = ("plastic", "paper", "cardboard", "metal", "glass", "organic")
# Target ids by config order: plastic=0 ... organic=5.
DET_NAMES = ("AluCan", "Glass", "PET", "CLOTH")
DET_MAPPING = {"AluCan": "metal", "Glass": "glass", "PET": "plastic", "CLOTH": "DROP"}


def make_spec(
    name: str = "src",
    annotation_type: str = "cls",
    mapping: dict[str, str] | None = None,
    drops: tuple[str, ...] = (),
) -> SourceSpec:
    return SourceSpec(
        name=name,
        fetcher=FetcherSpec(kind="local", ref="/tmp/none.zip", sha256=None),
        license="MIT",
        attribution="Fixture",
        annotation_type=annotation_type,
        background="clean",
        mapping=mapping or {"plastic": "plastic", "paper": "paper", "trash": "DROP"},
        drops=drops,
    )


def write(root: Path, rel: str, content: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def tree_digest(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


@pytest.fixture
def roots(tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    interim = tmp_path / "interim"
    raw.mkdir()
    return raw, interim


def test_flat_class_folders_and_drop_routing(roots: tuple[Path, Path]) -> None:
    raw, interim = roots
    write(raw, "src/plastic/a.jpg", "A")
    write(raw, "src/Paper/b.PNG", "B")  # case-insensitive folder + extension
    write(raw, "src/trash/c.jpg", "C")  # DROP -> wilderness
    write(raw, "src/notes.csv", "x")  # skipped, recorded
    before = tree_digest(raw)

    manifest = remap_source(make_spec(), raw, interim, TARGETS)

    assert (interim / "remapped/plastic/src__a.jpg").read_text() == "A"
    assert (interim / "remapped/paper/src__b.PNG").read_text() == "B"
    assert (interim / "wilderness/src__c.jpg").read_text() == "C"
    assert manifest.class_counts == {
        "plastic": 1,
        "paper": 1,
        "cardboard": 0,
        "metal": 0,
        "glass": 0,
        "organic": 0,
    }
    assert manifest.drop_count == 1
    assert manifest.skipped == ("notes.csv",)
    assert manifest.errors == ()
    assert tree_digest(raw) == before  # raw is never mutated
    assert RemapManifest.read(manifest_path(interim, "src")) == manifest


def test_nested_split_class_images_layout(roots: tuple[Path, Path]) -> None:
    raw, interim = roots
    write(raw, "src/train/Plastic/images/a.jpg", "A")
    write(raw, "src/val/plastic/b.jpg", "B")
    write(raw, "src/train/trash/images/c.jpg", "C")

    manifest = remap_source(make_spec(), raw, interim, TARGETS)

    assert (interim / "remapped/plastic/src__a.jpg").is_file()
    assert (interim / "remapped/plastic/src__b.jpg").is_file()
    assert (interim / "wilderness/src__c.jpg").is_file()
    assert manifest.class_counts["plastic"] == 2
    assert manifest.drop_count == 1


def test_collision_same_name_across_splits(roots: tuple[Path, Path]) -> None:
    raw, interim = roots
    write(raw, "src/train/plastic/x.jpg", "TRAIN")
    write(raw, "src/val/plastic/x.jpg", "VAL")

    manifest = remap_source(make_spec(), raw, interim, TARGETS)

    out = sorted((interim / "remapped/plastic").iterdir())
    assert [p.name for p in out] == ["src__x.jpg", "src__x__1.jpg"]
    assert {p.read_text() for p in out} == {"TRAIN", "VAL"}  # both copies kept
    assert manifest.class_counts["plastic"] == 2


def test_unmapped_class_folder_is_hard_error(roots: tuple[Path, Path]) -> None:
    raw, interim = roots
    write(raw, "src/plastic/a.jpg", "A")
    write(raw, "src/rubber/images/b.jpg", "B")

    with pytest.raises(UnmappedLabelError) as exc:
        remap_source(make_spec(), raw, interim, TARGETS)
    assert "rubber" in str(exc.value) and "src" in str(exc.value)
    assert not (interim / "remapped").exists()  # plan failed before any copy


def det_raw(raw: Path) -> None:
    write(raw, "src/data.yaml", "names: [AluCan, Glass, PET, CLOTH]\nnc: 4\n")
    write(raw, "src/images/train/a.jpg", "A")
    # AluCan -> metal(3), PET -> plastic(0); majority tie -> first kept box (metal)
    write(raw, "src/labels/train/a.txt", "0 0.5 0.5 0.2 0.2\n2 0.1 0.2 0.3 0.4\n")
    write(raw, "src/images/train/b.jpg", "B")  # CLOTH-only -> wilderness, box dropped
    write(raw, "src/labels/train/b.txt", "3 0.5 0.5 0.9 0.9\n")
    write(raw, "src/images/train/c.jpg", "C")  # Glass + dropped CLOTH box
    write(raw, "src/labels/train/c.txt", "1 0.4 0.4 0.1 0.1\n3 0.6 0.6 0.2 0.2\n")


def test_detection_label_rewrite_hand_computed(roots: tuple[Path, Path]) -> None:
    raw, interim = roots
    det_raw(raw)
    spec = make_spec(annotation_type="det", mapping=DET_MAPPING)
    before = tree_digest(raw)

    manifest = remap_source(spec, raw, interim, TARGETS)

    a_txt = interim / "remapped/metal/src__a.txt"
    assert a_txt.read_text() == "3 0.5 0.5 0.2 0.2\n0 0.1 0.2 0.3 0.4\n"
    assert (interim / "remapped/metal/src__a.jpg").read_text() == "A"
    assert (interim / "remapped/glass/src__c.txt").read_text() == "4 0.4 0.4 0.1 0.1\n"
    assert (interim / "wilderness/src__b.jpg").is_file()
    assert not (interim / "wilderness/src__b.txt").exists()  # no labels in the probe pool
    assert manifest.class_counts["metal"] == 1
    assert manifest.class_counts["glass"] == 1
    assert manifest.drop_count == 1
    assert manifest.dropped_boxes == 2  # b.txt CLOTH + c.txt CLOTH
    assert tree_digest(raw) == before


def test_detection_explicit_names_beat_discovery(roots: tuple[Path, Path]) -> None:
    raw, interim = roots
    write(raw, "src/images/a.jpg", "A")
    write(raw, "src/labels/a.txt", "1 0.5 0.5 0.2 0.2\n")  # names[1] = PET -> plastic(0)
    spec = make_spec(annotation_type="det", mapping=DET_MAPPING)

    manifest = remap_source(spec, raw, interim, TARGETS, names=("Glass", "PET"))

    assert (interim / "remapped/plastic/src__a.txt").read_text() == "0 0.5 0.5 0.2 0.2\n"
    assert manifest.class_counts["plastic"] == 1


def test_detection_classes_txt_fallback_and_missing_names(roots: tuple[Path, Path]) -> None:
    raw, interim = roots
    write(raw, "src/images/a.jpg", "A")
    write(raw, "src/labels/a.txt", "0 0.5 0.5 0.2 0.2\n")
    spec = make_spec(annotation_type="det", mapping=DET_MAPPING)

    with pytest.raises(ClassNamesError) as exc:  # no names, no data.yaml, no classes.txt
        remap_source(spec, raw, interim, TARGETS)
    assert "src" in str(exc.value)

    write(raw, "src/classes.txt", "Glass\nPET\n")
    manifest = remap_source(spec, raw, interim, TARGETS)
    assert (interim / "remapped/glass/src__a.txt").read_text() == "4 0.5 0.5 0.2 0.2\n"
    assert manifest.class_counts["glass"] == 1


def test_detection_unmapped_label_is_hard_error(roots: tuple[Path, Path]) -> None:
    raw, interim = roots
    write(raw, "src/data.yaml", "names: [rubber]\n")
    write(raw, "src/images/a.jpg", "A")
    write(raw, "src/labels/a.txt", "0 0.5 0.5 0.2 0.2\n")
    spec = make_spec(annotation_type="det", mapping=DET_MAPPING)

    with pytest.raises(UnmappedLabelError) as exc:
        remap_source(spec, raw, interim, TARGETS)
    assert "rubber" in str(exc.value) and "src" in str(exc.value)


def test_detection_missing_label_file_recorded(roots: tuple[Path, Path]) -> None:
    raw, interim = roots
    write(raw, "src/data.yaml", "names: [AluCan]\n")
    write(raw, "src/images/a.jpg", "A")  # no labels/a.txt anywhere

    manifest = remap_source(
        make_spec(annotation_type="det", mapping={"AluCan": "metal"}), raw, interim, TARGETS
    )

    assert manifest.errors == ("images/a.jpg: no YOLO label file found",)
    assert manifest.class_counts["metal"] == 0
    assert not (interim / "remapped/metal").exists()


def test_rerun_is_idempotent(roots: tuple[Path, Path]) -> None:
    raw, interim = roots
    write(raw, "src/plastic/a.jpg", "A")
    write(raw, "src/trash/c.jpg", "C")

    first = remap_source(make_spec(), raw, interim, TARGETS)
    second = remap_source(make_spec(), raw, interim, TARGETS)

    assert first == second
    assert [p.name for p in (interim / "remapped/plastic").iterdir()] == ["src__a.jpg"]
    assert [p.name for p in (interim / "wilderness").iterdir()] == ["src__c.jpg"]


def test_raw_tree_byte_identical_after_remap(roots: tuple[Path, Path]) -> None:
    raw, interim = roots
    write(raw, "src/plastic/a.jpg", "A")
    write(raw, "src/trash/c.jpg", "C")
    write(raw, "src/.manifest.json", '{"fetched_at": "x", "sha256": "y", "file_count": 2}')
    before = tree_digest(raw)

    remap_source(make_spec(), raw, interim, TARGETS)

    assert tree_digest(raw) == before
