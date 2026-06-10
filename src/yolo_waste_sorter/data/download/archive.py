"""Checksums + safe archive extraction.

data/raw/ is append-only: extraction NEVER overwrites an existing file, and
member paths are confined to the destination (no zip-slip).
"""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import zipfile
from pathlib import Path

from yolo_waste_sorter.data.download.errors import DownloadError

_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file, streamed."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def extract_archive(archive: Path, dest: Path) -> int:
    """Extract a zip/tar archive into dest, skipping members that already exist.

    Returns the number of files newly written. Unsupported formats fail fast.
    """
    dest.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        return _extract_zip(archive, dest)
    if tarfile.is_tarfile(archive):
        return _extract_tar(archive, dest)
    raise DownloadError(f"unsupported archive format: '{archive.name}' (expected zip or tar)")


def count_files(root: Path, exclude_names: frozenset[str] = frozenset()) -> int:
    """Number of regular files under root, excluding the given basenames."""
    return sum(1 for p in root.rglob("*") if p.is_file() and p.name not in exclude_names)


def _safe_target(dest: Path, member_name: str) -> Path:
    target = dest / member_name
    if not target.resolve().is_relative_to(dest.resolve()):
        raise DownloadError(f"archive member escapes destination: '{member_name}'")
    return target


def _extract_zip(archive: Path, dest: Path) -> int:
    written = 0
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            target = _safe_target(dest, info.filename)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists():
                continue  # append-only: never overwrite
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            written += 1
    return written


def _extract_tar(archive: Path, dest: Path) -> int:
    written = 0
    with tarfile.open(archive) as tf:
        for member in tf:
            target = _safe_target(dest, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise DownloadError(
                    f"refusing special tar member '{member.name}' (links/devices not allowed)"
                )
            if target.exists():
                continue  # append-only: never overwrite
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = tf.extractfile(member)
            if extracted is None:  # pragma: no cover - isfile() guarantees a stream
                raise DownloadError(f"could not read tar member '{member.name}'")
            with extracted as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            written += 1
    return written
