"""Review-queue copy semantics and the stratified sampling primitive."""

from pathlib import Path

import pytest
from helpers import prov, write_labels

from trashmonkey.data.qa import (
    FLAG_AREA_EXTREME,
    FLAG_CENTERBOX,
    FLAG_LOW_CONFIDENCE,
    emit_review_queue,
    run_checks,
    stratified_sample,
)


def test_review_queue_copies_and_indexes(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    write_labels(labels, "bad", [(0, 0.5, 0.5, 0.02, 0.02)])  # area_extreme
    write_labels(labels, "clean", [(0, 0.5, 0.5, 0.4, 0.4)])
    records = {
        "bad": prov(tmp_path, "bad", method="centerbox", confidence=0.1),
        "clean": prov(tmp_path, "clean", confidence=0.9),
    }
    report = run_checks(labels, records)
    out = tmp_path / "review"
    index = emit_review_queue(report, out)

    bad_flags = report.images["bad"].flags
    assert set(bad_flags) >= {FLAG_AREA_EXTREME, FLAG_LOW_CONFIDENCE, FLAG_CENTERBOX}
    for flag in bad_flags:  # copied into every flag dir it carries
        assert (out / flag / "bad.jpg").read_bytes() == b"fake-jpeg-bad"
        assert (out / flag / "bad.txt").exists()
    # originals untouched (copy, never move)
    assert Path(records["bad"].image).read_bytes() == b"fake-jpeg-bad"
    assert (labels / "bad.txt").exists()
    # clean image nowhere in the queue
    assert not list(out.rglob("clean*"))
    rows = index.read_text().splitlines()
    assert rows[0] == "image,flags,source,method,confidence"
    assert len(rows) == 2 and "bad.jpg" in rows[1] and FLAG_CENTERBOX in rows[1]


def test_review_queue_orders_by_ascending_confidence(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    records = {}
    for stem, conf in (("worst", 0.05), ("mid", 0.15), ("ok", 0.25)):
        write_labels(labels, stem, [(0, 0.5, 0.5, 0.4, 0.4)])
        records[stem] = prov(tmp_path, stem, confidence=conf)
    report = run_checks(labels, records)
    index = emit_review_queue(report, tmp_path / "review")
    stems = [row.split(",")[0] for row in index.read_text().splitlines()[1:]]
    assert [Path(s).stem for s in stems] == ["worst", "mid", "ok"]


def sample_items() -> list[dict[str, object]]:
    strata = [(0, "dino", 20), (1, "dino", 10), (1, "birefnet", 6), (2, "centerbox", 4)]
    return [
        {"class_id": cls, "method": method, "id": f"{cls}-{method}-{i}"}
        for cls, method, n in strata
        for i in range(n)
    ]


def test_stratified_sample_proportional_per_stratum() -> None:
    sample = stratified_sample(sample_items(), frac=0.5, seed=42)
    counts: dict[tuple[object, object], int] = {}
    for item in sample:
        key = (item["class_id"], item["method"])
        counts[key] = counts.get(key, 0) + 1
    assert counts == {(0, "dino"): 10, (1, "dino"): 5, (1, "birefnet"): 3, (2, "centerbox"): 2}


def test_stratified_sample_small_strata_always_represented() -> None:
    sample = stratified_sample(sample_items(), frac=0.1, seed=42)
    keys = {(item["class_id"], item["method"]) for item in sample}
    assert keys == {(0, "dino"), (1, "dino"), (1, "birefnet"), (2, "centerbox")}


def test_stratified_sample_deterministic_per_seed() -> None:
    items = sample_items()
    a = [i["id"] for i in stratified_sample(items, 0.5, seed=42)]
    assert a == [i["id"] for i in stratified_sample(items, 0.5, seed=42)]
    assert a != [i["id"] for i in stratified_sample(items, 0.5, seed=7)]


def test_stratified_sample_invalid_frac_rejected() -> None:
    for frac in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="frac"):
            stratified_sample(sample_items(), frac)
