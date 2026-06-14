"""Download-stage tests: registry schema, local fetcher, checksums, idempotency.

NO network: fake sources go through the `local` fetcher from a tmpdir.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import pytest

from trashmonkey.data.download import (
    MANIFEST_NAME,
    ChecksumMismatchError,
    DatasetConfigError,
    DownloadError,
    FetchError,
    FetcherSpec,
    SourceSpec,
    download_source,
    load_registry,
    parse_source,
)
from trashmonkey.data.download.archive import sha256_file
from trashmonkey.data.download.fetchers import fetch

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_CLASSES = ["plastic", "paper", "cardboard", "metal", "glass", "organic"]


def make_zip(tmp_path: Path, files: dict[str, str], name: str = "fake.zip") -> Path:
    archive = tmp_path / name
    with zipfile.ZipFile(archive, "w") as zf:
        for member, content in files.items():
            zf.writestr(member, content)
    return archive


def raw_entry(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": "fake",
        "fetcher": {"kind": "local", "ref": "/tmp/fake.zip", "sha256": None},
        "license": "MIT",
        "attribution": "Fake Source",
        "annotation_type": "cls",
        "background": "clean",
        "mapping": {"bottle": "plastic", "junk": "DROP"},
    }
    entry.update(overrides)
    return entry


def local_spec(archive: Path, sha256: str | None) -> SourceSpec:
    return parse_source(
        raw_entry(fetcher={"kind": "local", "ref": str(archive), "sha256": sha256}),
        TARGET_CLASSES,
    )


# --- registry schema -------------------------------------------------------


def test_real_registry_loads_with_trashnet() -> None:
    registry = load_registry(REPO_ROOT / "configs" / "datasets.yaml", TARGET_CLASSES)
    spec = registry["trashnet"]
    assert spec.fetcher.kind == "http"
    assert spec.license == "MIT"
    assert spec.annotation_type == "cls"
    assert spec.background == "clean"
    assert spec.dropped_labels() == {"trash"}
    assert set(spec.mapping.values()) == {"plastic", "paper", "cardboard", "metal", "glass", "DROP"}


def test_dominant_sources_carry_per_class_caps() -> None:
    # The two large sources cap their abundant classes at half the global
    # budget; organic (minority class) is left uncapped on both.
    registry = load_registry(REPO_ROOT / "configs" / "datasets.yaml", TARGET_CLASSES)
    capped = {"plastic": 750, "paper": 750, "cardboard": 750, "metal": 750, "glass": 750}
    for name in ("garbage-detection", "alistairking-household"):
        assert registry[name].cap == capped, name
        assert "organic" not in registry[name].cap, name
    # The clean priority sources stay uncapped.
    assert registry["trashnet"].cap == {}
    assert registry["drinking-waste"].cap == {}


def test_unknown_source_key_rejected_with_key_name() -> None:
    with pytest.raises(DatasetConfigError, match="unknown key 'flavor'"):
        parse_source(raw_entry(flavor="spicy"), TARGET_CLASSES)


def test_unknown_fetcher_key_rejected_with_key_name() -> None:
    bad = raw_entry(fetcher={"kind": "local", "ref": "/x", "sha256": None, "mirror": "y"})
    with pytest.raises(DatasetConfigError, match="unknown key 'mirror'"):
        parse_source(bad, TARGET_CLASSES)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"fetcher": {"kind": "ftp", "ref": "x"}}, "fetcher kind 'ftp'"),
        ({"annotation_type": "seg"}, "annotation_type 'seg'"),
        ({"background": "studio"}, "background 'studio'"),
        ({"mapping": {"bottle": "rest"}}, "mapping target 'rest'"),
        ({"mapping": {}}, "'mapping' must be a non-empty mapping"),
        ({"drops": ["bottle"]}, "listed in drops but mapped to 'plastic'"),
        ({"cap": {"plastic": 0}}, "must be a positive int"),
        ({"cap": {"rest": 5}}, "cap key 'rest'"),
        ({"fetcher": {"kind": "local", "ref": "/x", "sha256": "beef"}}, "64 lowercase hex"),
        ({"license": ""}, "'license' must be a non-empty string"),
    ],
)
def test_malformed_entries_rejected(overrides: dict[str, Any], match: str) -> None:
    with pytest.raises(DatasetConfigError, match=match):
        parse_source(raw_entry(**overrides), TARGET_CLASSES)


def test_duplicate_source_names_rejected(tmp_path: Path) -> None:
    config = tmp_path / "datasets.yaml"
    config.write_text(
        "sources:\n"
        + 2
        * (
            "  - name: dup\n"
            "    fetcher: {kind: local, ref: /x, sha256: null}\n"
            "    license: MIT\n"
            "    attribution: A\n"
            "    annotation_type: cls\n"
            "    background: clean\n"
            "    mapping: {a: plastic}\n"
        )
    )
    with pytest.raises(DatasetConfigError, match="duplicate source name 'dup'"):
        load_registry(config, TARGET_CLASSES)


# --- fetch + extract + manifest --------------------------------------------


def test_local_fetch_extracts_and_writes_manifest(tmp_path: Path) -> None:
    archive = make_zip(tmp_path, {"glass/a.jpg": "AA", "glass/b.jpg": "BB", "trash/c.jpg": "CC"})
    raw_root = tmp_path / "raw"
    result = download_source(local_spec(archive, sha256_file(archive)), raw_root)

    assert result.action == "fetched"
    assert result.file_count == 3
    assert (raw_root / "fake" / "glass" / "a.jpg").read_text() == "AA"
    manifest = json.loads((raw_root / "fake" / MANIFEST_NAME).read_text())
    assert manifest["sha256"] == result.sha256 == sha256_file(archive)
    assert manifest["file_count"] == 3
    assert manifest["fetched_at"]


def test_checksum_mismatch_fails_and_leaves_raw_clean(tmp_path: Path) -> None:
    archive = make_zip(tmp_path, {"glass/a.jpg": "AA"})
    raw_root = tmp_path / "raw"
    with pytest.raises(ChecksumMismatchError, match="sha256 mismatch for source 'fake'"):
        download_source(local_spec(archive, "0" * 64), raw_root)
    assert not (raw_root / "fake").exists()


def test_second_fetch_is_noop(tmp_path: Path) -> None:
    archive = make_zip(tmp_path, {"glass/a.jpg": "AA"})
    raw_root = tmp_path / "raw"
    spec = local_spec(archive, sha256_file(archive))

    first = download_source(spec, raw_root)
    manifest_before = (raw_root / "fake" / MANIFEST_NAME).read_bytes()
    second = download_source(spec, raw_root)

    assert (first.action, second.action) == ("fetched", "skipped")
    assert second.sha256 == first.sha256
    assert second.file_count == first.file_count
    assert (raw_root / "fake" / MANIFEST_NAME).read_bytes() == manifest_before


def test_null_sha256_still_skips_on_second_fetch(tmp_path: Path) -> None:
    archive = make_zip(tmp_path, {"glass/a.jpg": "AA"})
    spec = local_spec(archive, None)
    download_source(spec, tmp_path / "raw")
    assert download_source(spec, tmp_path / "raw").action == "skipped"


def test_force_refetches_but_never_overwrites(tmp_path: Path) -> None:
    archive = make_zip(tmp_path, {"glass/a.jpg": "AA", "glass/b.jpg": "BB"})
    raw_root = tmp_path / "raw"
    spec = local_spec(archive, sha256_file(archive))
    download_source(spec, raw_root)

    sentinel = raw_root / "fake" / "glass" / "a.jpg"
    sentinel.write_text("SENTINEL")  # test-only mutation to prove no-overwrite
    result = download_source(spec, raw_root, force=True)

    assert result.action == "fetched"
    assert sentinel.read_text() == "SENTINEL"


def test_stale_manifest_against_pinned_sha_fails(tmp_path: Path) -> None:
    archive = make_zip(tmp_path, {"glass/a.jpg": "AA"})
    raw_root = tmp_path / "raw"
    download_source(local_spec(archive, None), raw_root)
    with pytest.raises(ChecksumMismatchError, match="already present with sha256"):
        download_source(local_spec(archive, "f" * 64), raw_root)


def test_unsupported_archive_format_rejected(tmp_path: Path) -> None:
    blob = tmp_path / "data.bin"
    blob.write_text("not an archive")
    with pytest.raises(DownloadError, match="unsupported archive format"):
        download_source(local_spec(blob, None), tmp_path / "raw")


def test_missing_local_ref_fails(tmp_path: Path) -> None:
    with pytest.raises(FetchError, match="is not an existing file"):
        download_source(local_spec(tmp_path / "nope.zip", None), tmp_path / "raw")


# --- kaggle fetcher (no network, no CLI) ------------------------------------


def test_kaggle_missing_cli_fails_with_setup_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _cmd: None)
    spec = FetcherSpec(kind="kaggle", ref="someuser/somedataset", sha256=None)
    with pytest.raises(FetchError, match="kaggle CLI not found on PATH"):
        fetch(spec, tmp_path)
