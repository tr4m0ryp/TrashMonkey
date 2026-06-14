"""Tests for the T3 auto-boxing chain. Backends are injected fakes: no downloads, no GPU."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

from trashmonkey.data.autobox import (
    PROMPTS,
    BoxRecord,
    Detection,
    DinoPredictFn,
    MaskFn,
    box_directory,
    build_birefnet_backend,
    build_dino_backend,
    center_box,
    clamp_box,
    largest_component_box,
    mask_to_box,
    yolo_line,
)

HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
HAS_REMBG = importlib.util.find_spec("rembg") is not None


def make_image(directory: Path, name: str, width: int, height: int) -> Path:
    path = directory / name
    Image.new("RGB", (width, height), color=(200, 200, 200)).save(path)
    return path


def fake_dino(by_name: dict[str, list[Detection]]) -> DinoPredictFn:
    def predict(image_path: Path) -> Sequence[Detection]:
        return by_name[image_path.name]

    return predict


def fake_mask(by_name: dict[str, npt.NDArray[np.uint8]]) -> MaskFn:
    def predict(image_path: Path) -> npt.NDArray[np.uint8]:
        return by_name[image_path.name]

    return predict


def fail_mask(image_path: Path) -> npt.NDArray[np.uint8]:
    raise AssertionError(f"mask backend must not be called for {image_path}")


def blob_mask(height: int, width: int, rows: slice, cols: slice) -> npt.NDArray[np.uint8]:
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[rows, cols] = 255
    return mask


# --- geometry -----------------------------------------------------------------


def test_yolo_line_hand_computed() -> None:
    # 200x100 image, box (50,20)-(150,80): cx=100/200, cy=50/100, w=100/200, h=60/100.
    assert yolo_line(3, (50.0, 20.0, 150.0, 80.0), 200, 100) == (
        "3 0.500000 0.500000 0.500000 0.600000"
    )


def test_clamp_box_out_of_bounds_and_degenerate() -> None:
    assert clamp_box((-10.0, -5.0, 250.0, 120.0), 200, 100) == (0.0, 0.0, 200.0, 100.0)
    with pytest.raises(ValueError, match="degenerate"):
        clamp_box((300.0, 20.0, 350.0, 80.0), 200, 100)


def test_center_box_margin() -> None:
    assert center_box(200, 100, 0.05) == (10.0, 5.0, 190.0, 95.0)
    with pytest.raises(ValueError, match="margin"):
        center_box(200, 100, 0.5)


def test_largest_component_box_picks_bigger_blob() -> None:
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[0:10, 0:10] = 255  # 100 px
    mask[20:45, 20:45] = 255  # 625 px
    result = largest_component_box(mask)
    assert result is not None
    box, area = result
    assert box == (20.0, 20.0, 45.0, 45.0)
    assert area == 625


def test_largest_component_box_empty_mask() -> None:
    assert largest_component_box(np.zeros((10, 10), dtype=np.uint8)) is None


def test_mask_to_box_area_gates() -> None:
    tiny = blob_mask(100, 100, slice(0, 2), slice(0, 2))  # 0.04% of image
    full = np.full((100, 100), 255, dtype=np.uint8)  # 100% of image
    ok = blob_mask(100, 100, slice(20, 60), slice(10, 50))  # 16% of image
    assert mask_to_box(tiny, min_area_frac=0.05, max_area_frac=0.95) is None
    assert mask_to_box(full, min_area_frac=0.05, max_area_frac=0.95) is None
    assert mask_to_box(ok, min_area_frac=0.05, max_area_frac=0.95) == (10.0, 20.0, 50.0, 60.0)


# --- chain --------------------------------------------------------------------


def test_primary_dino_hit(tmp_path: Path) -> None:
    images = tmp_path / "imgs"
    images.mkdir()
    make_image(images, "a.png", 200, 100)
    records = box_directory(
        images,
        3,
        tmp_path / "labels",
        class_name="metal",
        source="trashnet",
        dino_predict=fake_dino({"a.png": [Detection((50.0, 20.0, 150.0, 80.0), 0.9)]}),
        birefnet_mask=fail_mask,
    )
    assert (tmp_path / "labels" / "a.txt").read_text() == "3 0.500000 0.500000 0.500000 0.600000\n"
    assert records == [BoxRecord("a.png", "trashnet", "dino", 0.9, [])]


def test_multibox_keeps_best_and_flags(tmp_path: Path) -> None:
    images = tmp_path / "imgs"
    images.mkdir()
    make_image(images, "a.png", 200, 100)
    detections = [
        Detection((0.0, 0.0, 20.0, 20.0), 0.4),
        Detection((50.0, 20.0, 150.0, 80.0), 0.8),
    ]
    (record,) = box_directory(
        images,
        0,
        tmp_path / "labels",
        class_name="plastic",
        source="trashnet",
        dino_predict=fake_dino({"a.png": detections}),
        birefnet_mask=fail_mask,
    )
    assert record.method == "dino"
    assert record.confidence == 0.8
    assert record.flags == ["multibox"]
    assert (tmp_path / "labels" / "a.txt").read_text() == "0 0.500000 0.500000 0.500000 0.600000\n"


def test_fallback_birefnet_when_dino_empty(tmp_path: Path) -> None:
    images = tmp_path / "imgs"
    images.mkdir()
    make_image(images, "a.png", 100, 80)
    mask = blob_mask(80, 100, slice(20, 60), slice(10, 50))  # 20% of image
    (record,) = box_directory(
        images,
        2,
        tmp_path / "labels",
        class_name="cardboard",
        source="taco",
        dino_predict=fake_dino({"a.png": []}),
        birefnet_mask=fake_mask({"a.png": mask}),
    )
    # rect (10,20)-(50,60) in 100x80: cx=0.3, cy=0.5, w=0.4, h=0.5
    assert (tmp_path / "labels" / "a.txt").read_text() == "2 0.300000 0.500000 0.400000 0.500000\n"
    assert record.method == "birefnet"
    assert record.confidence is None
    assert record.flags == []


def test_fallback_when_all_detections_below_threshold(tmp_path: Path) -> None:
    images = tmp_path / "imgs"
    images.mkdir()
    make_image(images, "a.png", 100, 80)
    mask = blob_mask(80, 100, slice(20, 60), slice(10, 50))
    (record,) = box_directory(
        images,
        1,
        tmp_path / "labels",
        class_name="paper",
        source="trashnet",
        min_confidence=0.5,
        dino_predict=fake_dino({"a.png": [Detection((10.0, 10.0, 90.0, 70.0), 0.4)]}),
        birefnet_mask=fake_mask({"a.png": mask}),
    )
    assert record.method == "birefnet"
    assert record.flags == []  # multibox never set on the fallback path


@pytest.mark.parametrize("mask_kind", ["empty", "tiny", "full"])
def test_last_resort_centerbox(tmp_path: Path, mask_kind: str) -> None:
    images = tmp_path / "imgs"
    images.mkdir()
    make_image(images, "a.png", 200, 100)
    masks = {
        "empty": np.zeros((100, 200), dtype=np.uint8),
        "tiny": blob_mask(100, 200, slice(0, 3), slice(0, 3)),
        "full": np.full((100, 200), 255, dtype=np.uint8),
    }
    (record,) = box_directory(
        images,
        5,
        tmp_path / "labels",
        class_name="organic",
        source="trashnet",
        dino_predict=fake_dino({"a.png": []}),
        birefnet_mask=fake_mask({"a.png": masks[mask_kind]}),
    )
    assert (tmp_path / "labels" / "a.txt").read_text() == "5 0.500000 0.500000 0.900000 0.900000\n"
    assert record.method == "centerbox"
    assert record.confidence is None
    assert record.flags == ["centerbox"]


def test_provenance_jsonl_and_flag_propagation(tmp_path: Path) -> None:
    images = tmp_path / "imgs"
    images.mkdir()
    for name in ("a.png", "b.png", "c.png"):
        make_image(images, name, 100, 80)
    dino = fake_dino(
        {
            "a.png": [
                Detection((10.0, 10.0, 50.0, 50.0), 0.9),
                Detection((0.0, 0.0, 9.0, 9.0), 0.35),
            ],
            "b.png": [],
            "c.png": [],
        }
    )
    mask = fake_mask(
        {
            "b.png": blob_mask(80, 100, slice(20, 60), slice(10, 50)),
            "c.png": np.zeros((80, 100), dtype=np.uint8),
        }
    )
    records = box_directory(
        images,
        4,
        tmp_path / "labels",
        class_name="glass",
        source="glassset",
        dino_predict=dino,
        birefnet_mask=mask,
    )
    lines = (tmp_path / "labels" / "provenance.jsonl").read_text().splitlines()
    parsed = [json.loads(line) for line in lines]
    assert [r["image"] for r in parsed] == ["a.png", "b.png", "c.png"]  # sorted order
    assert [r["method"] for r in parsed] == ["dino", "birefnet", "centerbox"]
    assert [r["flags"] for r in parsed] == [["multibox"], [], ["centerbox"]]
    assert all(r["source"] == "glassset" for r in parsed)
    assert parsed[0]["confidence"] == 0.9
    assert parsed[1]["confidence"] is None
    assert [r.to_dict() for r in records] == parsed


def test_progress_callback(tmp_path: Path) -> None:
    images = tmp_path / "imgs"
    images.mkdir()
    for name in ("a.png", "b.png"):
        make_image(images, name, 50, 50)
    seen: list[tuple[int, int, str]] = []
    box_directory(
        images,
        0,
        tmp_path / "labels",
        class_name="plastic",
        source="s",
        dino_predict=fake_dino(
            {n: [Detection((5.0, 5.0, 45.0, 45.0), 0.9)] for n in ("a.png", "b.png")}
        ),
        birefnet_mask=fail_mask,
        progress=lambda done, total, path: seen.append((done, total, path.name)),
    )
    assert seen == [(1, 2, "a.png"), (2, 2, "b.png")]


def test_unknown_class_name_rejected(tmp_path: Path) -> None:
    (tmp_path / "imgs").mkdir()
    with pytest.raises(ValueError, match="unknown class"):
        box_directory(tmp_path / "imgs", 0, tmp_path / "labels", class_name="rest", source="s")


def test_prompts_cover_all_six_classes() -> None:
    assert sorted(PROMPTS) == ["cardboard", "glass", "metal", "organic", "paper", "plastic"]


@pytest.mark.skipif(HAS_AUTODISTILL, reason="autodistill installed; would load real weights")
def test_dino_backend_import_error_names_extra() -> None:
    with pytest.raises(ImportError, match="boxing"):
        build_dino_backend(PROMPTS["plastic"])


@pytest.mark.skipif(HAS_REMBG, reason="rembg installed; would load real weights")
def test_birefnet_backend_import_error_names_extra() -> None:
    with pytest.raises(ImportError, match="boxing"):
        build_birefnet_backend()
