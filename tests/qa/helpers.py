"""Shared fixtures-as-functions for the QA test suite."""

import math
from pathlib import Path

from yolo_waste_sorter.data.qa import ProvenanceRecord, QAReport, run_checks

BoxTuple = tuple[int, float, float, float, float]


def write_labels(labels_dir: Path, stem: str, boxes: list[BoxTuple]) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"{c} {cx} {cy} {w} {h}" for c, cx, cy, w, h in boxes]
    (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def prov(
    tmp: Path, stem: str, method: str = "dino", confidence: float = 0.9, source: str = "trashnet"
) -> ProvenanceRecord:
    img = tmp / "images" / f"{stem}.jpg"
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"fake-jpeg-" + stem.encode())
    return ProvenanceRecord(str(img), source, method, confidence)


def check_one(tmp: Path, boxes: list[BoxTuple], **kwargs: object) -> list[str]:
    """Run checks on a single-image dataset, return its flags."""
    labels = tmp / "labels"
    write_labels(labels, "img", boxes)
    record = prov(tmp, "img", **kwargs)  # type: ignore[arg-type]
    return run_checks(labels, {"img": record}).images["img"].flags


def check_many(tmp: Path, boxes: dict[str, BoxTuple]) -> QAReport:
    """Run checks on a multi-image single-box-per-image dataset."""
    labels = tmp / "labels"
    records = {}
    for stem, box in boxes.items():
        write_labels(labels, stem, [box])
        records[stem] = prov(tmp, stem)
    return run_checks(labels, records)


def square(area: float) -> tuple[float, float]:
    return math.sqrt(area), math.sqrt(area)


def aspect_box(area: float, aspect: float) -> tuple[float, float]:
    return math.sqrt(area * aspect), math.sqrt(area / aspect)
