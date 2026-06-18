"""Split-stage tests (T6): LOSO TEST-1, instance grouping, stratified val,
dataset emission, and end-to-end determinism of the dedup->balance->split chain.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from item_helpers import make_items

from trashmonkey.data.balance import balance_items
from trashmonkey.data.dedup import Item, NearEdge, dedup_items, scan_remapped
from trashmonkey.data.split import SplitError, emit_dataset, split_items

CLASSES = ("plastic", "paper", "cardboard", "metal", "glass", "organic")
PRIORITY = ["trashnet", "gc3", "realwaste"]


def write_corpus(root: Path, class_name: str, source: str, n: int, seed0: int) -> list[Item]:
    items = []
    for i in range(n):
        path = root / class_name / f"{source}__{i:04d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed0 + i)
        Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(path)
        label = path.with_suffix(".txt")
        label.write_text(f"{CLASSES.index(class_name)} 0.5 0.5 0.5 0.5\n")
        items.append(
            Item(
                key=f"{class_name}/{path.name}",
                class_name=class_name,
                source=source,
                image=path,
                label=label,
            )
        )
    return items


def test_loso_test1_excludes_source_from_train_val() -> None:
    items = make_items("glass", "gc3", 40) + make_items("glass", "realwaste", 25)
    result = split_items(items, [], leave_out_source="realwaste", val_fraction=0.2)
    for item in items:
        expected_test = item.source == "realwaste"
        assert (result.assignments[item.key] == "test") == expected_test
    assert result.counts["test"] == {"glass": 25}


def test_null_leave_out_source_warns_and_skips_test1(
    caplog: pytest.LogCaptureFixture,
) -> None:
    items = make_items("glass", "gc3", 20)
    with caplog.at_level(logging.WARNING, logger="trashmonkey.data.split"):
        result = split_items(items, [], leave_out_source=None, val_fraction=0.2)
    assert "test" not in result.counts
    assert set(result.assignments.values()) <= {"train", "val"}
    assert any("TEST-1" in rec.message for rec in caplog.records)


def test_missing_leave_out_source_raises() -> None:
    items = make_items("glass", "gc3", 5)
    with pytest.raises(SplitError, match="realwaste"):
        split_items(items, [], leave_out_source="realwaste", val_fraction=0.2)


def test_groups_never_straddle_split_boundaries() -> None:
    items = make_items("paper", "trashnet", 30) + make_items("paper", "gc3", 30)
    edges = [
        NearEdge("paper/trashnet__0000.png", "paper/gc3__0000.png", 4),
        NearEdge("paper/gc3__0000.png", "paper/gc3__0001.png", 5),
        NearEdge("paper/trashnet__0010.png", "paper/trashnet__0011.png", 3),
    ]
    result = split_items(items, edges, leave_out_source=None, val_fraction=0.3)
    for edge in edges:
        assert result.assignments[edge.key_a] == result.assignments[edge.key_b]
        assert result.group_ids[edge.key_a] == result.group_ids[edge.key_b]
    chain = ["paper/trashnet__0000.png", "paper/gc3__0000.png", "paper/gc3__0001.png"]
    assert result.group_ids[chain[0]] == min(chain)


def test_val_fraction_default_and_stratified_counts() -> None:
    items = make_items("metal", "gc3", 100)
    result = split_items(items, [], leave_out_source=None, val_fraction=None)
    assert result.val_fraction == 0.15
    assert result.counts["val"]["metal"] == 15
    assert result.counts["train"]["metal"] == 85
    explicit = split_items(items, [], leave_out_source=None, val_fraction=0.2)
    assert explicit.counts["val"]["metal"] == 20


def test_val_fraction_bounds() -> None:
    items = make_items("metal", "gc3", 10)
    with pytest.raises(SplitError, match="val_fraction"):
        split_items(items, [], leave_out_source=None, val_fraction=1.5)


def test_wild_test_holds_exactly_test_only_sources() -> None:
    items = (
        make_items("glass", "gc3", 30)
        + make_items("glass", "garbage", 12)
        + make_items("glass", "realwaste", 8)
    )
    result = split_items(
        items,
        [],
        leave_out_source="realwaste",
        val_fraction=0.2,
        test_only_sources={"garbage"},
    )
    for item in items:
        split = result.assignments[item.key]
        if item.source == "garbage":
            assert split == "wild_test"
        elif item.source == "realwaste":
            assert split == "test"
        else:
            assert split in {"train", "val"}
    assert result.counts["wild_test"] == {"glass": 12}
    # train/val exclude both the leave-out and the test_only source.
    trainval = {k for k, v in result.assignments.items() if v in {"train", "val"}}
    assert all("garbage__" not in k and "realwaste__" not in k for k in trainval)
    assert result.test_only_sources == ("garbage",)


def test_clean_test_carves_stratified_fraction_without_straddling() -> None:
    items = make_items("metal", "trashnet", 40) + make_items("metal", "gc3", 40)
    edges = [
        # two trashnet images form one physical-object group.
        NearEdge("metal/trashnet__0000.png", "metal/trashnet__0001.png", 4),
    ]
    result = split_items(
        items,
        edges,
        leave_out_source=None,
        val_fraction=0.2,
        clean_holdout_sources={"trashnet"},
        clean_holdout_fraction=0.25,
    )
    clean = {k for k, v in result.assignments.items() if v == "clean_test"}
    # only the trashnet source is eligible; gc3 is never carved.
    assert all("trashnet__" in k for k in clean)
    assert all(result.assignments[k] in {"train", "val"} for k in items_keys(items, "gc3"))
    # 40 trashnet images -> 39 groups (one pair) -> round(0.25*39)=10 groups.
    trashnet_groups = {result.group_ids[k] for k in items_keys(items, "trashnet")}
    carved_groups = {result.group_ids[k] for k in clean}
    assert len(carved_groups) == round(0.25 * len(trashnet_groups))
    # the near-dup pair never straddles: both members share one split.
    a, b = "metal/trashnet__0000.png", "metal/trashnet__0001.png"
    assert result.assignments[a] == result.assignments[b]
    assert result.clean_holdout_fraction == 0.25
    assert result.clean_holdout_sources == ("trashnet",)


def items_keys(items: list[Item], source: str) -> list[str]:
    return [it.key for it in items if it.source == source]


def test_inert_knobs_equal_legacy_three_split() -> None:
    items = make_items("glass", "gc3", 40) + make_items("glass", "realwaste", 25)
    edges = [NearEdge("glass/gc3__0000.png", "glass/gc3__0001.png", 4)]
    legacy = split_items(items, edges, leave_out_source="realwaste", val_fraction=0.2)
    with_knobs = split_items(
        items,
        edges,
        leave_out_source="realwaste",
        val_fraction=0.2,
        test_only_sources=frozenset(),
        clean_holdout_sources=frozenset(),
        clean_holdout_fraction=0.0,
    )
    assert with_knobs.assignments == legacy.assignments
    assert with_knobs.counts == legacy.counts
    assert set(legacy.assignments.values()) == {"train", "val", "test"}


def test_emit_dataset_layout_and_yaml(tmp_path: Path) -> None:
    root = tmp_path / "remapped"
    items = write_corpus(root, "plastic", "gc3", 10, 100)
    items += write_corpus(root, "plastic", "realwaste", 4, 900)
    result = split_items(items, [], leave_out_source="realwaste", val_fraction=0.2)
    yaml_path = emit_dataset(result, items, tmp_path / "processed", "baseline", CLASSES)
    dataset_root = tmp_path / "processed" / "baseline"
    assert yaml_path == dataset_root / "dataset.yaml"
    spec = yaml.safe_load(yaml_path.read_text())
    assert spec["path"] == str(dataset_root.resolve())
    assert spec["train"] == "images/train"
    assert spec["val"] == "images/val"
    assert spec["test"] == "images/test"
    assert spec["names"] == {i: c for i, c in enumerate(CLASSES)}
    test_images = sorted(p.name for p in (dataset_root / "images" / "test").iterdir())
    assert test_images == [f"plastic__realwaste__{i:04d}.png" for i in range(4)]
    for split in ("train", "val", "test"):
        images = sorted((dataset_root / "images" / split).iterdir())
        labels = sorted((dataset_root / "labels" / split).iterdir())
        assert [p.stem for p in images] == [p.stem for p in labels]
        assert all(p.suffix == ".txt" for p in labels)
    assert labels[0].read_text().startswith("0 ")


def test_emit_includes_clean_and_wild_test_splits(tmp_path: Path) -> None:
    root = tmp_path / "remapped"
    items = write_corpus(root, "plastic", "trashnet", 12, 100)
    items += write_corpus(root, "plastic", "garbage", 4, 500)
    items += write_corpus(root, "plastic", "realwaste", 4, 900)
    result = split_items(
        items,
        [],
        leave_out_source="realwaste",
        val_fraction=0.2,
        test_only_sources={"garbage"},
        clean_holdout_sources={"trashnet"},
        clean_holdout_fraction=0.25,
    )
    yaml_path = emit_dataset(result, items, tmp_path / "processed", "baseline", CLASSES)
    spec = yaml.safe_load(yaml_path.read_text())
    # every emitted split has members; emission order follows SPLITS.
    assert list(spec)[:1] == ["path"]
    for split in ("train", "val", "test", "clean_test", "wild_test"):
        assert spec[split] == f"images/{split}"
    dataset_root = tmp_path / "processed" / "baseline"
    wild = sorted(p.name for p in (dataset_root / "images" / "wild_test").iterdir())
    assert wild == [f"plastic__garbage__{i:04d}.png" for i in range(4)]
    clean_names = [p.name for p in (dataset_root / "images" / "clean_test").iterdir()]
    assert clean_names and all("trashnet__" in n for n in clean_names)


def test_emit_omits_empty_new_splits(tmp_path: Path) -> None:
    root = tmp_path / "remapped"
    items = write_corpus(root, "plastic", "gc3", 8, 100)
    items += write_corpus(root, "plastic", "realwaste", 4, 900)
    result = split_items(items, [], leave_out_source="realwaste", val_fraction=0.2)
    yaml_path = emit_dataset(result, items, tmp_path / "processed", "baseline", CLASSES)
    spec = yaml.safe_load(yaml_path.read_text())
    assert "clean_test" not in spec and "wild_test" not in spec
    assert set(spec) == {"path", "train", "val", "test", "names"}


def test_emit_requires_labels(tmp_path: Path) -> None:
    items = make_items("glass", "gc3", 3)
    result = split_items(items, [], leave_out_source=None, val_fraction=0.3)
    with pytest.raises(SplitError, match="no YOLO label"):
        emit_dataset(result, items, tmp_path / "processed", "baseline", CLASSES)


def run_chain(remapped: Path, out: Path) -> None:
    items = scan_remapped(remapped, ["plastic", "glass"])
    deduped = dedup_items(items, PRIORITY)
    deduped.write_manifest(out / "dedup.yaml")
    balanced = balance_items(
        deduped.kept, cap=8, floor=2, exempt_sources=frozenset({"realwaste"})
    )
    balanced.write_manifest(out / "balance.yaml")
    result = split_items(
        balanced.kept, deduped.near_edges, leave_out_source="realwaste", val_fraction=0.25
    )
    result.write_manifest(out / "split.yaml")
    emit_dataset(result, balanced.kept, out / "processed", "baseline", CLASSES)


def test_chain_is_deterministic_end_to_end(tmp_path: Path) -> None:
    root = tmp_path / "remapped"
    write_corpus(root, "plastic", "trashnet", 6, 0)
    write_corpus(root, "plastic", "gc3", 12, 50)
    write_corpus(root, "glass", "gc3", 10, 200)
    write_corpus(root, "glass", "realwaste", 5, 300)
    # one exact cross-source duplicate: same pixels as trashnet's plastic 0000
    dup = root / "plastic" / "gc3__dup.png"
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(dup)
    dup.with_suffix(".txt").write_text("0 0.5 0.5 0.5 0.5\n")

    run_chain(root, tmp_path / "run1")
    run_chain(root, tmp_path / "run2")
    for name in ("dedup.yaml", "balance.yaml", "split.yaml"):
        assert (tmp_path / "run1" / name).read_bytes() == (
            tmp_path / "run2" / name
        ).read_bytes(), f"{name} differs between identical runs"
    dedup = yaml.safe_load((tmp_path / "run1" / "dedup.yaml").read_text())
    assert dedup["dropped"] == [
        {"image": "plastic/gc3__dup.png", "duplicate_of": "plastic/trashnet__0000.png",
         "distance": 0}
    ]
    assert dedup["overlap_matrix"] == {"trashnet": {"gc3": 1}}
    split = yaml.safe_load((tmp_path / "run1" / "split.yaml").read_text())
    assert split["counts"]["test"] == {"glass": 5}
    yaml1 = (tmp_path / "run1" / "processed" / "baseline" / "dataset.yaml").read_text()
    yaml2 = (tmp_path / "run2" / "processed" / "baseline" / "dataset.yaml").read_text()
    assert yaml1.replace("run1", "run2") == yaml2
