"""Balance-stage tests (T4): caps, stratification, floor warning, determinism.

Items are constructed in memory -- balancing never touches the filesystem.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from yolo_waste_sorter.data.balance import balance_items
from yolo_waste_sorter.data.dedup import Item


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


def test_cap_applies_and_never_duplicates() -> None:
    items = (
        make_items("plastic", "trashnet", 50)
        + make_items("plastic", "gc3", 30)
        + make_items("plastic", "drinking", 20)
    )
    result = balance_items(items, cap=60, floor=10)
    keys = [it.key for it in result.kept]
    assert len(keys) == 60
    assert len(set(keys)) == 60  # without replacement: no duplicates, ever
    per_source = result.counts["plastic"]
    assert per_source["trashnet"]["kept"] == 30  # proportional 60 * 50/100
    assert per_source["gc3"]["kept"] == 18
    assert per_source["drinking"]["kept"] == 12
    assert per_source["trashnet"]["dropped"] == 20


def test_no_source_wiped_out_of_a_class() -> None:
    items = make_items("metal", "gc3", 1000) + make_items("metal", "trashnet", 3)
    result = balance_items(items, cap=100, floor=10)
    assert result.counts["metal"]["trashnet"]["kept"] >= 1
    assert sum(c["kept"] for c in result.counts["metal"].values()) == 100


def test_small_class_kept_whole() -> None:
    items = make_items("organic", "gc3", 40)
    result = balance_items(items, cap=1500, floor=10)
    assert len(result.kept) == 40
    assert result.counts["organic"]["gc3"] == {"kept": 40, "dropped": 0}


def test_floor_warning_logged_not_raised(caplog: pytest.LogCaptureFixture) -> None:
    items = make_items("organic", "gc3", 12)
    with caplog.at_level(logging.WARNING, logger="yolo_waste_sorter.data.balance"):
        result = balance_items(items, cap=1500, floor=800)
    assert len(result.kept) == 12  # warning, never an error or a drop
    assert any("organic" in w and "800" in w for w in result.floor_warnings)
    assert any("floor" in rec.message for rec in caplog.records)


def test_exempt_source_bypasses_cap() -> None:
    items = make_items("glass", "realwaste", 30) + make_items("glass", "gc3", 50)
    result = balance_items(items, cap=10, floor=1, exempt_sources=frozenset({"realwaste"}))
    kept_realwaste = [it for it in result.kept if it.source == "realwaste"]
    kept_gc3 = [it for it in result.kept if it.source == "gc3"]
    assert len(kept_realwaste) == 30  # ALL of the TEST-1 source survives
    assert len(kept_gc3) == 10
    assert "realwaste" not in result.counts.get("glass", {})


def test_per_source_caps_clamp_before_global_cap() -> None:
    items = make_items("paper", "gc3", 50) + make_items("paper", "trashnet", 50)
    result = balance_items(items, cap=100, floor=1, source_caps={"gc3": {"paper": 5}})
    assert result.counts["paper"]["gc3"]["kept"] == 5
    assert result.counts["paper"]["trashnet"]["kept"] == 50


def test_sampling_is_seeded_and_deterministic() -> None:
    items = make_items("plastic", "gc3", 200)
    first = balance_items(items, cap=50, floor=1, seed=42)
    second = balance_items(list(reversed(items)), cap=50, floor=1, seed=42)
    assert [it.key for it in first.kept] == [it.key for it in second.kept]
    other_seed = balance_items(items, cap=50, floor=1, seed=43)
    assert [it.key for it in first.kept] != [it.key for it in other_seed.kept]


def test_manifest_roundtrip_deterministic(tmp_path: Path) -> None:
    items = make_items("plastic", "gc3", 30) + make_items("metal", "trashnet", 5)
    balance_items(items, cap=20, floor=3).write_manifest(tmp_path / "m1.yaml")
    balance_items(items, cap=20, floor=3).write_manifest(tmp_path / "m2.yaml")
    assert (tmp_path / "m1.yaml").read_bytes() == (tmp_path / "m2.yaml").read_bytes()
