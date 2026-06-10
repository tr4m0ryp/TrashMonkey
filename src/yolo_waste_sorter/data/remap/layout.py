"""Interim output layout: collision-free destinations, copies, idempotent cleanup.

data/raw/ is read-only here -- files are only ever copied out of it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from yolo_waste_sorter.data.remap.manifest import (
    MANIFESTS_DIRNAME,
    REMAPPED_DIRNAME,
    WILDERNESS_DIRNAME,
    manifest_path,
)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


class DestAllocator:
    """Allocates destination paths prefixed `<source>__`; disambiguates collisions.

    Same-named originals (e.g. train/x.jpg and val/x.jpg) get a deterministic
    `__<n>` tag. When a YOLO label rides along, the matching .txt stem is
    reserved too, so image/label pairs never split.
    """

    def __init__(self, source: str) -> None:
        self._source = source
        self._taken: set[Path] = set()

    def allocate(self, directory: Path, orig_name: str, *, with_label: bool = False) -> Path:
        stem = Path(orig_name).stem
        suffix = Path(orig_name).suffix
        attempt = 0
        while True:
            tag = "" if attempt == 0 else f"__{attempt}"
            dest = directory / f"{self._source}__{stem}{tag}{suffix}"
            clashes = {dest, dest.with_suffix(".txt")} if with_label else {dest}
            if not any(c in self._taken or c.exists() for c in clashes):
                self._taken.update(clashes)
                return dest
            attempt += 1


def clean_previous(interim_root: Path, source: str) -> None:
    """Remove this source's outputs from a previous run so re-runs are idempotent."""
    prefix = f"{source}__"
    remapped = interim_root / REMAPPED_DIRNAME
    if remapped.is_dir():
        for class_dir in remapped.iterdir():
            if not class_dir.is_dir() or class_dir.name == MANIFESTS_DIRNAME:
                continue
            for entry in class_dir.iterdir():
                if entry.is_file() and entry.name.startswith(prefix):
                    entry.unlink()
    wilderness = interim_root / WILDERNESS_DIRNAME
    if wilderness.is_dir():
        for entry in wilderness.iterdir():
            if entry.is_file() and entry.name.startswith(prefix):
                entry.unlink()
    mpath = manifest_path(interim_root, source)
    if mpath.exists():
        mpath.unlink()


def copy_into(src: Path, dest: Path) -> None:
    """Copy (never move) one raw file to its interim destination."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
