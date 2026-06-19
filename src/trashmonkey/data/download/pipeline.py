"""Download orchestration: fetch -> verify sha256 -> extract -> manifest.

Idempotent: a source whose manifest matches the registry checksum is skipped;
`force=True` refetches (extraction still never overwrites existing files in
the append-only data/raw/). Per-source state lives in
data/raw/<source>/.manifest.json and feeds the pipeline's resume logic.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from trashmonkey.data.download.archive import count_files, extract_archive, sha256_file
from trashmonkey.data.download.errors import ChecksumMismatchError, DownloadError
from trashmonkey.data.download.fetchers import fetch
from trashmonkey.data.download.registry import SourceSpec

MANIFEST_NAME = ".manifest.json"


@dataclass(frozen=True)
class Manifest:
    """Per-source fetch record, stored at data/raw/<source>/.manifest.json."""

    fetched_at: str
    sha256: str
    file_count: int

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def read(cls, path: Path) -> Manifest:
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict) or set(raw) != {"fetched_at", "sha256", "file_count"}:
            raise DownloadError(f"malformed manifest at {path}: {raw!r}")
        return cls(
            fetched_at=str(raw["fetched_at"]),
            sha256=str(raw["sha256"]),
            file_count=int(raw["file_count"]),
        )


@dataclass(frozen=True)
class DownloadResult:
    """Outcome of download_source for one registry entry."""

    source: str
    action: str  # "fetched" | "skipped"
    sha256: str
    file_count: int
    dest: Path


def download_source(spec: SourceSpec, raw_root: Path, *, force: bool = False) -> DownloadResult:
    """Materialize one source under raw_root/<name>/; skip if already present."""
    dest = raw_root / spec.name
    manifest_path = dest / MANIFEST_NAME
    expected = spec.fetcher.sha256

    if manifest_path.exists() and not force:
        manifest = Manifest.read(manifest_path)
        if expected is not None and manifest.sha256 != expected:
            raise ChecksumMismatchError(
                f"source '{spec.name}' already present with sha256 {manifest.sha256}, but the "
                f"registry expects {expected}. data/raw/ is append-only -- resolve the registry "
                f"entry or refetch with --force after moving the stale copy aside."
            )
        return DownloadResult(spec.name, "skipped", manifest.sha256, manifest.file_count, dest)

    with tempfile.TemporaryDirectory(prefix=f"yws-fetch-{spec.name}-") as tmp:
        archive = fetch(spec.fetcher, Path(tmp))
        digest = sha256_file(archive)
        if expected is not None and digest != expected:
            raise ChecksumMismatchError(
                f"sha256 mismatch for source '{spec.name}': expected {expected}, got {digest}. "
                f"Archive discarded; data/raw/ untouched."
            )
        extract_archive(archive, dest)

    manifest = Manifest(
        fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
        sha256=digest,
        file_count=count_files(dest, exclude_names=frozenset({MANIFEST_NAME})),
    )
    manifest.write(manifest_path)
    return DownloadResult(spec.name, "fetched", digest, manifest.file_count, dest)


def download_sources(
    specs: Iterable[SourceSpec], raw_root: Path, *, force: bool = False
) -> list[DownloadResult]:
    """Download every given source in order; fail fast on the first error.

    Deliberately SEQUENTIAL: the kaggle CLI is not concurrency-safe -- it creates
    its config dir with a non-atomic exists-then-makedirs, so parallel kaggle
    fetches race and one crashes with FileExistsError on a fresh VM. The big
    sources are kaggle, so parallelism bought little anyway.
    """
    return [download_source(spec, raw_root, force=force) for spec in specs]
