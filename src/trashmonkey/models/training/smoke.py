"""Synthetic smoke dataset: PIL-drawn colored shapes on white, YOLO layout.

Two images per class (12 total for the six waste classes), each holding one
shape whose bounds are the ground-truth box. White background mirrors the
deployment presentation. train and val point at the same 12 images -- the
smoke run only proves the full cycle executes, never measures anything.
"""

from __future__ import annotations

import random
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

IMAGES_PER_CLASS = 2

_PALETTE: tuple[tuple[int, int, int], ...] = (
    (220, 40, 40),
    (40, 90, 220),
    (150, 100, 40),
    (120, 120, 130),
    (40, 170, 90),
    (240, 160, 30),
)


def build_smoke_dataset(
    classes: tuple[str, ...],
    root: Path,
    imgsz: int = 160,
    images_per_class: int = IMAGES_PER_CLASS,
    seed: int = 42,
) -> Path:
    """Write the dataset under ``root`` and return the data.yaml path."""
    image_dir = root / "images" / "train"
    label_dir = root / "labels" / "train"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    for class_id, class_name in enumerate(classes):
        color = _PALETTE[class_id % len(_PALETTE)]
        for index in range(images_per_class):
            canvas = Image.new("RGB", (imgsz, imgsz), (255, 255, 255))
            draw = ImageDraw.Draw(canvas)
            width = rng.randint(imgsz // 4, imgsz // 2)
            height = rng.randint(imgsz // 4, imgsz // 2)
            x0 = rng.randint(4, imgsz - width - 4)
            y0 = rng.randint(4, imgsz - height - 4)
            box = (x0, y0, x0 + width, y0 + height)
            if class_id % 2 == 0:
                draw.rectangle(box, fill=color)
            else:
                draw.ellipse(box, fill=color)
            stem = f"{class_name}_{index:02d}"
            canvas.save(image_dir / f"{stem}.jpg", quality=90)
            center_x = (x0 + width / 2) / imgsz
            center_y = (y0 + height / 2) / imgsz
            label = f"{class_id} {center_x:.6f} {center_y:.6f} {width / imgsz:.6f} {height / imgsz:.6f}\n"
            (label_dir / f"{stem}.txt").write_text(label)

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root.resolve()),
                "train": "images/train",
                "val": "images/train",
                "names": dict(enumerate(classes)),
            },
            sort_keys=False,
        )
    )
    return data_yaml
