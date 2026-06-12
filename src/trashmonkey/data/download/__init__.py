"""Dataset download stage: source registry, fetchers, checksums, extraction.

Driven by configs/datasets.yaml; archives land append-only in data/raw/<source>/
with a .manifest.json per source. Public surface re-exported here.
"""

from trashmonkey.data.download.errors import (
    ChecksumMismatchError,
    DatasetConfigError,
    DownloadError,
    FetchError,
)
from trashmonkey.data.download.pipeline import (
    MANIFEST_NAME,
    DownloadResult,
    Manifest,
    download_source,
    download_sources,
)
from trashmonkey.data.download.registry import (
    DROP,
    FetcherSpec,
    SourceSpec,
    load_registry,
    parse_source,
)

__all__ = [
    "DROP",
    "MANIFEST_NAME",
    "ChecksumMismatchError",
    "DatasetConfigError",
    "DownloadError",
    "DownloadResult",
    "FetchError",
    "FetcherSpec",
    "Manifest",
    "SourceSpec",
    "download_source",
    "download_sources",
    "load_registry",
    "parse_source",
]
