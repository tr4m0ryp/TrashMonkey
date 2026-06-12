"""Dedup-stage tests: interim scan, pHash exact-dup priority, near-dup edges.

Tiny synthetic PIL images only -- no network, no GPU. Band-boundary tests
inject a deterministic hash_fn; the real imagehash path is covered by the
identical-image tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from trashmonkey.data.dedup import (
    DedupError,
    Item,
    dedup_items,
    scan_remapped,
)

PRIORITY = ["trashnet", "gc3", "realwaste"]


def noise_image(path: Path, seed: int, size: int = 64) -> None:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def make_item(root: Path, class_name: str, source: str, name: str, seed: int) -> Item:
    path = root / class_name / f"{source}__{name}.png"
    noise_image(path, seed)
    return Item(
        key=f"{class_name}/{path.name}",
        class_name=class_name,
        source=source,
        image=path,
        label=None,
    )


def test_scan_remapped_layout(tmp_path: Path) -> None:
    root = tmp_path / "remapped"
    noise_image(root / "plastic" / "trashnet__a.png", 1)
    noise_image(root / "glass" / "gc3__b.jpg", 2)
    (root / "glass" / "gc3__b.txt").write_text("4 0.5 0.5 0.4 0.4\n")
    (root / "plastic" / "notes.csv").write_text("ignored\n")
    items = scan_remapped(root, ["plastic", "glass"])
    assert [it.key for it in items] == ["plastic/trashnet__a.png", "glass/gc3__b.jpg"]
    assert items[0].source == "trashnet" and items[0].label is None
    assert items[1].source == "gc3" and items[1].label is not None


def test_scan_rejects_bad_filename_and_missing_dir(tmp_path: Path) -> None:
    bad = tmp_path / "bad-name"
    noise_image(bad / "plastic" / "no-separator.png", 1)
    with pytest.raises(DedupError, match="<source>__<name>"):
        scan_remapped(bad, ["plastic"])
    incomplete = tmp_path / "no-glass"
    noise_image(incomplete / "plastic" / "trashnet__a.png", 1)
    with pytest.raises(DedupError, match="missing class directory"):
        scan_remapped(incomplete, ["plastic", "glass"])


def test_exact_dup_drops_later_source(tmp_path: Path) -> None:
    a = make_item(tmp_path, "glass", "trashnet", "x", seed=7)
    b = make_item(tmp_path, "glass", "gc3", "y", seed=7)  # identical pixels
    c = make_item(tmp_path, "metal", "gc3", "z", seed=8)  # unrelated
    result = dedup_items([a, b, c], PRIORITY)
    assert {it.key for it in result.kept} == {a.key, c.key}
    assert len(result.dropped) == 1
    assert result.dropped[0].key == b.key
    assert result.dropped[0].duplicate_of == a.key
    assert result.overlap_matrix == {"trashnet": {"gc3": 1}}


def test_exact_dup_priority_is_registry_order(tmp_path: Path) -> None:
    a = make_item(tmp_path, "glass", "trashnet", "x", seed=7)
    b = make_item(tmp_path, "glass", "gc3", "y", seed=7)
    result = dedup_items([a, b], ["gc3", "trashnet"])
    assert [it.key for it in result.kept] == [b.key]
    assert result.dropped[0].key == a.key
    assert result.overlap_matrix == {"gc3": {"trashnet": 1}}


def test_unknown_source_raises(tmp_path: Path) -> None:
    a = make_item(tmp_path, "glass", "mystery", "x", seed=1)
    with pytest.raises(DedupError, match="mystery"):
        dedup_items([a], PRIORITY)


def test_near_dup_band_records_edges(tmp_path: Path) -> None:
    a = make_item(tmp_path, "paper", "trashnet", "a", seed=1)
    b = make_item(tmp_path, "paper", "gc3", "b", seed=2)  # distance 3 -> near edge
    c = make_item(tmp_path, "paper", "gc3", "c", seed=3)  # distance 8 -> near edge
    d = make_item(tmp_path, "paper", "realwaste", "d", seed=4)  # distance >8 to all
    hashes = {a.image: 0, b.image: 0b111, c.image: 0xFF, d.image: 0xFFFFF}
    result = dedup_items([a, b, c, d], PRIORITY, hash_fn=lambda p: hashes[p])
    assert len(result.kept) == 4 and not result.dropped
    pairs = {(e.key_a, e.key_b): e.distance for e in result.near_edges}
    assert pairs[(a.key, b.key)] == 3
    assert pairs[(a.key, c.key)] == 8
    assert all(d.key not in pair for pair in pairs)
    assert result.near_matrix["trashnet"] == {"gc3": 2}


def test_near_band_lower_bound_is_exact_dup(tmp_path: Path) -> None:
    a = make_item(tmp_path, "paper", "trashnet", "a", seed=1)
    b = make_item(tmp_path, "paper", "gc3", "b", seed=2)
    hashes = {a.image: 0, b.image: 0b11}  # distance 2 -> exact duplicate
    result = dedup_items([a, b], PRIORITY, hash_fn=lambda p: hashes[p])
    assert [it.key for it in result.kept] == [a.key]
    assert result.dropped[0].distance == 2
    assert not result.near_edges


def test_dedup_manifest_deterministic(tmp_path: Path) -> None:
    items = [
        make_item(tmp_path, "glass", "trashnet", "x", seed=7),
        make_item(tmp_path, "glass", "gc3", "y", seed=7),
        make_item(tmp_path, "metal", "realwaste", "z", seed=9),
    ]
    first = dedup_items(items, PRIORITY)
    second = dedup_items(list(reversed(items)), PRIORITY)
    assert first.to_dict() == second.to_dict()
    first.write_manifest(tmp_path / "m1.yaml")
    second.write_manifest(tmp_path / "m2.yaml")
    assert (tmp_path / "m1.yaml").read_bytes() == (tmp_path / "m2.yaml").read_bytes()
