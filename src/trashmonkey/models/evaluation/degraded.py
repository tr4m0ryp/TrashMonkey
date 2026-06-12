"""Materialize degraded split copies for TEST-2 metrics and tuner dumps (T6/T9).

Each severity level gets ``<work_dir>/severity_<s>/{images,labels}/<split>/``:
images run through the shared ``degrade_image`` camera model (deterministic in
``(image, severity, seed)``, written as lossless PNG so bytes are reproducible)
while label files are copied unchanged. A per-severity ``dataset.yaml`` points
ultralytics val at the degraded copies.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from trashmonkey.models.evaluation.report import EvalError
from trashmonkey.utils.degrade import degrade_image

IMAGE_SUFFIXES = (".bmp", ".jpeg", ".jpg", ".png", ".webp")


def load_dataset_spec(data_yaml: Path) -> dict[str, Any]:
    """Read a YOLO dataset yaml; fail fast on a missing root or names block."""
    if not data_yaml.is_file():
        raise EvalError(f"dataset yaml not found: {data_yaml}")
    with open(data_yaml) as f:
        spec = yaml.safe_load(f)
    if not isinstance(spec, dict) or "path" not in spec or "names" not in spec:
        raise EvalError(f"{data_yaml}: expected a mapping with 'path' and 'names' keys")
    return spec


def split_images(data_yaml: Path, split: str) -> list[Path]:
    """Sorted image files of one split of the dataset behind ``data_yaml``."""
    spec = load_dataset_spec(data_yaml)
    if split not in spec:
        raise EvalError(f"{data_yaml}: split {split!r} is not defined")
    image_dir = Path(spec["path"]) / str(spec[split])
    if not image_dir.is_dir():
        raise EvalError(f"split {split!r} image directory not found: {image_dir}")
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise EvalError(f"split {split!r} has no images under {image_dir}")
    return images


def _degrade_file(src: Path, dest: Path, severity: int, seed: int) -> None:
    bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if bgr is None:
        raise EvalError(f"cannot read image: {src}")
    rgb = np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
    degraded = degrade_image(rgb, severity, seed)
    out = cv2.cvtColor(degraded, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(dest), out):
        raise EvalError(f"cannot write degraded image: {dest}")


def materialize_severity(
    data_yaml: Path, splits: tuple[str, ...], severity: int, seed: int, work_dir: Path
) -> Path:
    """Write degraded copies of ``splits`` at one severity; returns the yaml.

    Labels are copied verbatim (degradation never moves a box). The emitted
    yaml carries a 'train' key only because ultralytics ``check_det_dataset``
    requires it -- nothing ever trains on these copies.
    """
    spec = load_dataset_spec(data_yaml)
    root = work_dir / f"severity_{severity}"
    label_root = Path(spec["path"]) / "labels"
    for split in splits:
        images = split_images(data_yaml, split)
        image_dest = root / "images" / split
        label_dest = root / "labels" / split
        image_dest.mkdir(parents=True, exist_ok=True)
        label_dest.mkdir(parents=True, exist_ok=True)
        for src in images:
            _degrade_file(src, image_dest / (src.stem + ".png"), severity, seed)
            label_src = label_root / split / (src.stem + ".txt")
            if not label_src.is_file():
                raise EvalError(f"label file missing for {src.name}: {label_src}")
            shutil.copy2(label_src, label_dest / label_src.name)

    out_spec: dict[str, Any] = {"path": str(root.resolve())}
    out_spec["train"] = f"images/{splits[0]}"  # required key, never used
    for split in splits:
        out_spec[split] = f"images/{split}"
    out_spec.setdefault("val", out_spec[splits[0]])
    out_spec["names"] = spec["names"]
    yaml_path = root / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(out_spec, f, sort_keys=False)
    return yaml_path
