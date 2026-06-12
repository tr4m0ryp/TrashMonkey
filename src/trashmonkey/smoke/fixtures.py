"""Deterministic synthetic fixture archives for the smoke harness (task 016).

PIL-drawn colored shapes on a white background, seed 42 -- never real dataset
content. Each source becomes one byte-stable zip (fixed member order and
timestamps) consumed by the download framework's ``local`` fetcher. Generation
fails fast if any two images land within the dedup near-duplicate band
(pHash Hamming distance <= 8), so the smoke pipeline counts stay deterministic.
"""

from __future__ import annotations

import io
import random
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

SEED = 42
IMAGE_SIZE = 224
ARCHIVE_NAMES = ("alpha.zip", "beta.zip")
MIN_PHASH_DISTANCE = 9  # strictly above the dedup NEAR_BAND upper bound (8)

# Source label -> image count. alpha covers all six classes plus the DROP
# label "junk" (wilderness routing); beta is the TEST-1 leave-out source.
ALPHA_LABELS: dict[str, int] = {
    "bottle": 4,
    "sheet": 4,
    "box": 4,
    "can": 4,
    "jar": 4,
    "peel": 4,
    "junk": 4,
}
BETA_LABELS: dict[str, int] = {"jar": 2, "peel": 2}

_COLORS: dict[str, tuple[int, int, int]] = {
    "bottle": (210, 60, 60),
    "sheet": (60, 90, 210),
    "box": (150, 100, 40),
    "can": (95, 95, 110),
    "jar": (40, 175, 165),
    "peel": (90, 170, 50),
    "junk": (200, 60, 180),
}
_SHAPES = ("ellipse", "rectangle", "triangle", "diamond")


class FixtureError(Exception):
    """Fixture generation produced unusable (near-duplicate) images."""


def _draw_shape(draw: ImageDraw.ImageDraw, shape: str, box: tuple[float, float, float, float],
                color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = box
    if shape == "ellipse":
        draw.ellipse(box, fill=color)
    elif shape == "rectangle":
        draw.rectangle(box, fill=color)
    elif shape == "triangle":
        draw.polygon([((x1 + x2) / 2, y1), (x1, y2), (x2, y2)], fill=color)
    else:  # diamond
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        draw.polygon([(cx, y1), (x2, cy), (cx, y2), (x1, cy)], fill=color)


def render_image(source: str, label: str, index: int) -> bytes:
    """One synthetic PNG: a colored shape plus confetti, deterministic in args."""
    rng = random.Random(f"{SEED}:{source}:{label}:{index}")
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    half = IMAGE_SIZE * (0.16 + 0.14 * rng.random())
    cx = IMAGE_SIZE * (0.30 + 0.40 * rng.random())
    cy = IMAGE_SIZE * (0.30 + 0.40 * rng.random())
    _draw_shape(
        draw,
        _SHAPES[index % len(_SHAPES)],
        (cx - half, cy - half, cx + half, cy + half),
        _COLORS[label],
    )
    palette = list(_COLORS.values())
    for _ in range(6):  # confetti decorrelates the perceptual hashes
        r = 3 + 5 * rng.random()
        dx = IMAGE_SIZE * rng.random()
        dy = IMAGE_SIZE * rng.random()
        draw.ellipse((dx - r, dy - r, dx + r, dy + r), fill=rng.choice(palette))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _members(source: str, labels: dict[str, int]) -> dict[str, bytes]:
    return {
        f"{label}/{label}{index:02d}.png": render_image(source, label, index)
        for label in sorted(labels)
        for index in range(labels[label])
    }


def _check_phash_separation(blobs: dict[str, bytes]) -> int:
    """Min pairwise pHash distance; raises inside the dedup near-dup band."""
    import imagehash

    hashes = {
        name: int(str(imagehash.phash(Image.open(io.BytesIO(blob)))), 16)
        for name, blob in blobs.items()
    }
    names = sorted(hashes)
    minimum = 64
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            distance = (hashes[a] ^ hashes[b]).bit_count()
            if distance < MIN_PHASH_DISTANCE:
                raise FixtureError(
                    f"fixture images {a!r} and {b!r} are pHash distance {distance} apart "
                    f"(< {MIN_PHASH_DISTANCE}): the dedup stage would merge them -- "
                    "adjust the generator seed/shapes"
                )
            minimum = min(minimum, distance)
    return minimum


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    """Byte-stable zip: sorted members, fixed timestamps, stored (PNG is compressed)."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            zf.writestr(info, members[name])


def generate_fixtures(dest_dir: Path) -> dict[str, int]:
    """(Re)write alpha.zip + beta.zip under ``dest_dir``; returns image counts."""
    by_source = {"alpha": _members("alpha", ALPHA_LABELS), "beta": _members("beta", BETA_LABELS)}
    flat = {
        f"{source}/{name}": blob
        for source, members in by_source.items()
        for name, blob in members.items()
    }
    _check_phash_separation(flat)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for source, members in by_source.items():
        _write_zip(dest_dir / f"{source}.zip", members)
    return {source: len(members) for source, members in by_source.items()}
