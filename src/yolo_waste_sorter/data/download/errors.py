"""Download-stage exceptions. Fail fast -- no silent fallbacks."""


class DownloadError(Exception):
    """Base class for download-stage failures."""


class DatasetConfigError(DownloadError):
    """configs/datasets.yaml violates the source-registry schema."""


class FetchError(DownloadError):
    """A fetcher could not produce the source archive."""


class ChecksumMismatchError(DownloadError):
    """Archive sha256 does not match the registry entry."""
