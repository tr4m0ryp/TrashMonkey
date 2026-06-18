"""Tests for the pure label-quality filter (T4).

Covers each drop path (centerbox method, low confidence, whole-frame box,
speck box), the keep paths (tight DINO box, det source with no provenance,
defensive ambiguity), reason recording, and determinism.
"""

from pathlib import Path

from trashmonkey.data.dedup import Item
from trashmonkey.data.qa.report import ProvenanceRecord
from trashmonkey.data.quality import (
    REASON_BOX_TOO_LARGE,
    REASON_BOX_TOO_SMALL,
    REASON_LOW_CONFIDENCE,
    REASON_METHOD,
    FilterResult,
    filter_items,
)

# Default thresholds used across tests (mirrors the QA layer's bars).
DROP_METHODS = frozenset({"centerbox"})
MIN_CONFIDENCE = 0.30
MAX_BOX_FRAC = 0.95
MIN_BOX_FRAC = 0.05


def _item(tmp_path: Path, stem: str, *, source: str, lines: list[str] | None) -> Item:
    """Build an Item, writing a YOLO label file when ``lines`` is given."""
    image = tmp_path / f"{stem}.jpg"
    label: Path | None = None
    if lines is not None:
        label = tmp_path / f"{stem}.txt"
        label.write_text("".join(f"{line}\n" for line in lines))
    return Item(
        key=f"plastic/{source}__{stem}.jpg",
        class_name="plastic",
        source=source,
        image=image,
        label=label,
    )


def _prov(stem: str, *, method: str, confidence: float, source: str = "dino_src") -> ProvenanceRecord:
    return ProvenanceRecord(
        image=f"{stem}.jpg", source=source, method=method, confidence=confidence
    )


def _run(items: list[Item], prov: dict[str, ProvenanceRecord]) -> FilterResult:
    return filter_items(
        items,
        prov,
        drop_methods=DROP_METHODS,
        min_confidence=MIN_CONFIDENCE,
        max_box_frac=MAX_BOX_FRAC,
        min_box_frac=MIN_BOX_FRAC,
    )


def test_centerbox_method_dropped(tmp_path: Path) -> None:
    item = _item(tmp_path, "a", source="trashnet", lines=["0 0.5 0.5 0.4 0.4"])
    prov = {"a": _prov("a", method="centerbox", confidence=0.9)}
    result = _run([item], prov)
    assert result.kept == ()
    assert result.dropped == (item,)
    assert result.reasons[item.key] == REASON_METHOD


def test_low_confidence_dropped(tmp_path: Path) -> None:
    item = _item(tmp_path, "b", source="trashnet", lines=["0 0.5 0.5 0.4 0.4"])
    prov = {"b": _prov("b", method="dino", confidence=0.10)}  # below 0.30
    result = _run([item], prov)
    assert result.dropped == (item,)
    assert result.reasons[item.key] == REASON_LOW_CONFIDENCE


def test_whole_frame_box_dropped(tmp_path: Path) -> None:
    # 0.99 * 0.99 = 0.9801 area fraction > 0.95.
    item = _item(tmp_path, "c", source="trashnet", lines=["0 0.5 0.5 0.99 0.99"])
    prov = {"c": _prov("c", method="dino", confidence=0.9)}
    result = _run([item], prov)
    assert result.dropped == (item,)
    assert result.reasons[item.key] == REASON_BOX_TOO_LARGE


def test_speck_box_dropped(tmp_path: Path) -> None:
    # 0.1 * 0.1 = 0.01 area fraction < 0.05.
    item = _item(tmp_path, "d", source="trashnet", lines=["0 0.5 0.5 0.1 0.1"])
    prov = {"d": _prov("d", method="dino", confidence=0.9)}
    result = _run([item], prov)
    assert result.dropped == (item,)
    assert result.reasons[item.key] == REASON_BOX_TOO_SMALL


def test_tight_dino_box_kept(tmp_path: Path) -> None:
    # 0.4 * 0.5 = 0.20 area fraction, comfortably mid-range; high confidence; dino.
    item = _item(tmp_path, "e", source="trashnet", lines=["0 0.5 0.5 0.4 0.5"])
    prov = {"e": _prov("e", method="dino", confidence=0.85)}
    result = _run([item], prov)
    assert result.kept == (item,)
    assert result.dropped == ()
    assert result.reasons == {}


def test_det_item_no_provenance_kept(tmp_path: Path) -> None:
    # Detector source never auto-boxed: no provenance entry, normal geometry.
    item = _item(tmp_path, "f", source="taco", lines=["0 0.5 0.5 0.3 0.3"])
    result = _run([item], {})
    assert result.kept == (item,)
    assert result.reasons == {}


def test_det_item_no_label_kept(tmp_path: Path) -> None:
    # No label path at all (label=None): nothing to geometry-judge, no provenance.
    item = _item(tmp_path, "g", source="taco", lines=None)
    result = _run([item], {})
    assert result.kept == (item,)


def test_det_item_degenerate_box_dropped(tmp_path: Path) -> None:
    # A det source's own shipped label IS dropped when clearly degenerate.
    item = _item(tmp_path, "h", source="taco", lines=["0 0.5 0.5 0.99 0.99"])
    result = _run([item], {})
    assert result.dropped == (item,)
    assert result.reasons[item.key] == REASON_BOX_TOO_LARGE


def test_largest_box_judged_for_multi_line(tmp_path: Path) -> None:
    # Two boxes: small speck + a mid box. Judge the LARGEST -> kept.
    item = _item(
        tmp_path, "i", source="taco", lines=["0 0.2 0.2 0.05 0.05", "0 0.6 0.6 0.4 0.4"]
    )
    result = _run([item], {})
    assert result.kept == (item,)


def test_largest_box_whole_frame_among_many_dropped(tmp_path: Path) -> None:
    # Largest box is whole-frame -> dropped even though a tight box also present.
    item = _item(
        tmp_path, "j", source="taco", lines=["0 0.5 0.5 0.3 0.3", "0 0.5 0.5 0.99 0.99"]
    )
    result = _run([item], {})
    assert result.reasons[item.key] == REASON_BOX_TOO_LARGE


def test_method_precedence_over_confidence(tmp_path: Path) -> None:
    # Both centerbox method and low confidence: method reason wins (precedence).
    item = _item(tmp_path, "k", source="trashnet", lines=["0 0.5 0.5 0.4 0.4"])
    prov = {"k": _prov("k", method="centerbox", confidence=0.05)}
    result = _run([item], prov)
    assert result.reasons[item.key] == REASON_METHOD


def test_confidence_precedence_over_geometry(tmp_path: Path) -> None:
    # Low confidence AND speck geometry: confidence reason wins (precedence).
    item = _item(tmp_path, "l", source="trashnet", lines=["0 0.5 0.5 0.1 0.1"])
    prov = {"l": _prov("l", method="dino", confidence=0.1)}
    result = _run([item], prov)
    assert result.reasons[item.key] == REASON_LOW_CONFIDENCE


def test_confidence_at_threshold_kept(tmp_path: Path) -> None:
    # Strictly-below bar: confidence == min_confidence is KEPT.
    item = _item(tmp_path, "m", source="trashnet", lines=["0 0.5 0.5 0.4 0.4"])
    prov = {"m": _prov("m", method="dino", confidence=MIN_CONFIDENCE)}
    result = _run([item], prov)
    assert result.kept == (item,)


def test_malformed_label_kept(tmp_path: Path) -> None:
    # Malformed line (wrong field count) -> ambiguous -> KEEP, no drop.
    item = _item(tmp_path, "n", source="taco", lines=["0 0.5 0.5"])
    result = _run([item], {})
    assert result.kept == (item,)


def test_out_of_range_label_kept(tmp_path: Path) -> None:
    # Out-of-range geometry -> ambiguous -> KEEP.
    item = _item(tmp_path, "o", source="taco", lines=["0 0.5 0.5 1.5 0.5"])
    result = _run([item], {})
    assert result.kept == (item,)


def test_empty_label_kept(tmp_path: Path) -> None:
    # Empty label file -> no parseable box -> not geometry-judged -> KEEP.
    item = _item(tmp_path, "p", source="taco", lines=[])
    result = _run([item], {})
    assert result.kept == (item,)


def test_missing_label_file_kept(tmp_path: Path) -> None:
    # Label path set but file absent on disk -> ambiguous -> KEEP.
    item = Item(
        key="plastic/taco__q.jpg",
        class_name="plastic",
        source="taco",
        image=tmp_path / "q.jpg",
        label=tmp_path / "q.txt",  # never written
    )
    result = _run([item], {})
    assert result.kept == (item,)


def test_box_frac_boundaries_kept(tmp_path: Path) -> None:
    # Exactly at max (0.95) and at min (0.05): inclusive bounds -> KEPT.
    # sqrt-ish picks: w*h == max/min exactly via square-ish factors.
    big = _item(tmp_path, "r", source="taco", lines=["0 0.5 0.5 0.95 1.0"])  # 0.95
    small = _item(tmp_path, "s", source="taco", lines=["0 0.5 0.5 0.05 1.0"])  # 0.05
    result = _run([big, small], {})
    assert set(result.kept) == {big, small}
    assert result.reasons == {}


def test_reasons_only_for_dropped(tmp_path: Path) -> None:
    keep = _item(tmp_path, "t", source="trashnet", lines=["0 0.5 0.5 0.4 0.4"])
    drop = _item(tmp_path, "u", source="trashnet", lines=["0 0.5 0.5 0.4 0.4"])
    prov = {
        "t": _prov("t", method="dino", confidence=0.9),
        "u": _prov("u", method="centerbox", confidence=0.9),
    }
    result = _run([keep, drop], prov)
    assert result.kept == (keep,)
    assert result.dropped == (drop,)
    assert set(result.reasons) == {drop.key}


def test_output_order_follows_input(tmp_path: Path) -> None:
    a = _item(tmp_path, "a1", source="taco", lines=["0 0.5 0.5 0.3 0.3"])
    b = _item(tmp_path, "b1", source="taco", lines=["0 0.5 0.5 0.3 0.3"])
    c = _item(tmp_path, "c1", source="taco", lines=["0 0.5 0.5 0.3 0.3"])
    result = _run([c, a, b], {})
    assert result.kept == (c, a, b)


def test_determinism_same_input_same_output(tmp_path: Path) -> None:
    items = [
        _item(tmp_path, "x", source="trashnet", lines=["0 0.5 0.5 0.4 0.4"]),
        _item(tmp_path, "y", source="trashnet", lines=["0 0.5 0.5 0.99 0.99"]),
        _item(tmp_path, "z", source="taco", lines=["0 0.5 0.5 0.3 0.3"]),
    ]
    prov = {
        "x": _prov("x", method="centerbox", confidence=0.9),
        "y": _prov("y", method="dino", confidence=0.9),
    }
    first = _run(items, prov)
    second = _run(items, prov)
    assert first.kept == second.kept
    assert first.dropped == second.dropped
    assert first.reasons == second.reasons
