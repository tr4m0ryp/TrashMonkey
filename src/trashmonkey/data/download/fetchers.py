"""Fetchers produce a source archive in a scratch directory.

They never touch data/raw/ -- the pipeline verifies the checksum first and
extracts afterwards. Kinds: kaggle (external CLI), http, local (path copy,
exists primarily for tests).
"""

from __future__ import annotations

import shutil
import ssl
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from trashmonkey.data.download.errors import FetchError
from trashmonkey.data.download.registry import FetcherSpec

KAGGLE_SETUP_HINT = (
    "Install the kaggle CLI (e.g. `pipx install kaggle`) and authenticate by placing an API "
    "token at ~/.kaggle/kaggle.json (kaggle.com -> Settings -> API -> Create New Token)."
)


def fetch(spec: FetcherSpec, scratch: Path) -> Path:
    """Fetch the archive described by `spec` into `scratch`; return its path."""
    if spec.kind == "kaggle":
        return _fetch_kaggle(spec, scratch)
    if spec.kind == "http":
        return _fetch_http(spec, scratch)
    if spec.kind == "local":
        return _fetch_local(spec, scratch)
    raise FetchError(f"unknown fetcher kind '{spec.kind}'")  # registry validation prevents this


def _fetch_kaggle(spec: FetcherSpec, scratch: Path) -> Path:
    exe = shutil.which("kaggle")
    if exe is None:
        raise FetchError(f"kaggle CLI not found on PATH. {KAGGLE_SETUP_HINT}")
    cmd = [exe, "datasets", "download", "-d", spec.ref, "-p", str(scratch)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        detail = (proc.stderr.strip() or proc.stdout.strip())[-500:]
        raise FetchError(
            f"`kaggle datasets download -d {spec.ref}` failed (exit {proc.returncode}): "
            f"{detail or '<no output>'}. If this is an authentication error: {KAGGLE_SETUP_HINT}"
        )
    archives = sorted(p for p in scratch.iterdir() if p.is_file())
    if not archives:
        raise FetchError(f"kaggle CLI reported success for '{spec.ref}' but produced no file")
    if len(archives) > 1:
        raise FetchError(f"kaggle download for '{spec.ref}' produced multiple files: {archives}")
    return archives[0]


def _ssl_context() -> ssl.SSLContext:
    """Default TLS context, sourcing CAs from certifi when the system store is unusable.

    Some local Python builds ship without a wired-up CA store (macOS framework
    builds); certifi is the canonical CA bundle in that situation, not a
    security downgrade.
    """
    context = ssl.create_default_context()
    if context.cert_store_stats().get("x509_ca", 0) == 0:
        try:
            import certifi
        except ImportError as exc:
            raise FetchError(
                "the Python SSL CA store is empty and certifi is not installed; "
                "run `pip install certifi` or fix the interpreter's certificates"
            ) from exc
        context = ssl.create_default_context(cafile=certifi.where())
    return context


def _fetch_http(spec: FetcherSpec, scratch: Path) -> Path:
    filename = Path(urllib.parse.urlparse(spec.ref).path).name or "archive.bin"
    target = scratch / filename
    request = urllib.request.Request(spec.ref, headers={"User-Agent": "trashmonkey/0.1"})
    try:
        with (
            urllib.request.urlopen(request, context=_ssl_context()) as response,
            open(target, "wb") as out,
        ):
            shutil.copyfileobj(response, out)
    except (urllib.error.URLError, OSError) as exc:
        raise FetchError(f"HTTP fetch of '{spec.ref}' failed: {exc}") from exc
    return target


def _fetch_local(spec: FetcherSpec, scratch: Path) -> Path:
    source = Path(spec.ref)
    if not source.is_file():
        raise FetchError(f"local fetcher ref '{spec.ref}' is not an existing file")
    target = scratch / source.name
    shutil.copy2(source, target)
    return target
