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

from yolo_waste_sorter.data.balance import balance_items
from yolo_waste_sorter.data.dedup import Item, NearEdge, dedup_items, scan_remapped
from yolo_waste_sorter.data.split import SplitError, emit_dataset, split_items

CLASSES = ("plastic", "paper", "cardboard", "metal", "glass", "organic")
PRIORITY = ["trashnet", "gc3", "realwaste"]


def make_items(class_name: str, source: str, n: int) -> list[Item]:
    return [
        Item(
            key=f"{class_name}/{source}__{i:04d}.png",
            class_name=class_name,
            source=source,
            image=Path(f"/fake/{class_name}/{source}__{i:04d}.png"),
            label=None,
        )
        for i in range(n)
    ]


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
    with caplog.at_level(logging.WARNING, logger="yolo_waste_sorter.data.split"):
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
