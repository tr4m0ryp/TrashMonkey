"""Schema + validation for configs/datasets.yaml (the source registry).

Per-source entry shape (other pipeline stages depend on it):
name, fetcher {kind: kaggle|http|local, ref, sha256}, license, attribution,
annotation_type (cls|det), background (clean|wild),
mapping {<source_label>: <target_class|DROP>}, drops (optional), cap (optional).
Validation is strict: unknown keys fail with the key name in the error.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from trashmonkey.data.download.errors import DatasetConfigError

DROP = "DROP"
FETCHER_KINDS = frozenset({"kaggle", "http", "local"})
ANNOTATION_TYPES = frozenset({"cls", "det"})
BACKGROUNDS = frozenset({"clean", "wild"})

_TOP_KEYS = frozenset({"sources"})
_FETCHER_KEYS = frozenset({"kind", "ref", "sha256"})
_SOURCE_KEYS = frozenset(
    {
        "name",
        "fetcher",
        "license",
        "attribution",
        "annotation_type",
        "background",
        "mapping",
        "drops",
        "cap",
        "class_names",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FetcherSpec:
    """How to obtain one source archive."""

    kind: str
    ref: str
    sha256: str | None


@dataclass(frozen=True)
class SourceSpec:
    """One validated entry of the source registry."""

    name: str
    fetcher: FetcherSpec
    license: str
    attribution: str
    annotation_type: str
    background: str
    mapping: dict[str, str]
    drops: tuple[str, ...] = ()
    cap: dict[str, int] = field(default_factory=dict)
    # Detection-source class-index order (index i -> this label), for sources
    # whose YOLO labels ship no data.yaml/classes.txt. Empty for cls sources or
    # det sources that carry their own names file.
    class_names: tuple[str, ...] = ()

    def dropped_labels(self) -> frozenset[str]:
        """Source labels excluded from training (mapping DROPs union `drops`)."""
        mapped = {label for label, target in self.mapping.items() if target == DROP}
        return frozenset(mapped | set(self.drops))


def _fail(context: str, message: str) -> DatasetConfigError:
    return DatasetConfigError(f"{context}: {message}")


def _check_keys(raw: dict[str, Any], allowed: frozenset[str], context: str) -> None:
    for key in raw:
        if key not in allowed:
            raise _fail(context, f"unknown key '{key}' (allowed: {', '.join(sorted(allowed))})")


def _str(raw: dict[str, Any], key: str, context: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise _fail(context, f"'{key}' must be a non-empty string, got {value!r}")
    return value


def _parse_fetcher(raw: Any, context: str) -> FetcherSpec:
    if not isinstance(raw, dict):
        raise _fail(context, f"'fetcher' must be a mapping, got {type(raw).__name__}")
    _check_keys(raw, _FETCHER_KEYS, context)
    kind = _str(raw, "kind", context)
    if kind not in FETCHER_KINDS:
        raise _fail(context, f"fetcher kind '{kind}' not in {sorted(FETCHER_KINDS)}")
    ref = _str(raw, "ref", context)
    sha256 = raw.get("sha256")
    if sha256 is not None and (not isinstance(sha256, str) or not _SHA256_RE.match(sha256)):
        raise _fail(context, f"sha256 must be null or 64 lowercase hex chars, got {sha256!r}")
    return FetcherSpec(kind=kind, ref=ref, sha256=sha256)


def _parse_mapping(raw: Any, targets: frozenset[str], context: str) -> dict[str, str]:
    if not isinstance(raw, dict) or not raw:
        raise _fail(context, "'mapping' must be a non-empty mapping of source label -> target")
    mapping: dict[str, str] = {}
    for label, target in raw.items():
        if not isinstance(label, str) or not isinstance(target, str):
            raise _fail(context, f"mapping entries must be str -> str, got {label!r}: {target!r}")
        if target != DROP and target not in targets:
            raise _fail(
                context,
                f"mapping target '{target}' for label '{label}' is not a known target class "
                f"({sorted(targets)}) or {DROP}",
            )
        mapping[label] = target
    return mapping


def _parse_drops(raw: Any, mapping: dict[str, str], context: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        raise _fail(context, f"'drops' must be a list of source labels, got {raw!r}")
    for label in raw:
        if mapping.get(label, DROP) != DROP:
            raise _fail(
                context,
                f"label '{label}' is listed in drops but mapped to '{mapping[label]}'",
            )
    return tuple(raw)


def _parse_cap(raw: Any, targets: frozenset[str], context: str) -> dict[str, int]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise _fail(context, f"'cap' must be a mapping of target class -> int, got {raw!r}")
    cap: dict[str, int] = {}
    for cls, limit in raw.items():
        if cls not in targets:
            raise _fail(context, f"cap key '{cls}' is not a target class ({sorted(targets)})")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise _fail(context, f"cap for '{cls}' must be a positive int, got {limit!r}")
        cap[cls] = limit
    return cap


def parse_source(raw: Any, target_classes: Collection[str]) -> SourceSpec:
    """Validate one raw registry entry; raise DatasetConfigError on any violation."""
    if not isinstance(raw, dict):
        raise _fail("source entry", f"must be a mapping, got {type(raw).__name__}")
    context = f"source '{raw.get('name', '<unnamed>')}'"
    _check_keys(raw, _SOURCE_KEYS, context)
    targets = frozenset(target_classes)
    name = _str(raw, "name", context)
    annotation_type = _str(raw, "annotation_type", context)
    if annotation_type not in ANNOTATION_TYPES:
        raise _fail(context, f"annotation_type '{annotation_type}' not in {sorted(ANNOTATION_TYPES)}")
    background = _str(raw, "background", context)
    if background not in BACKGROUNDS:
        raise _fail(context, f"background '{background}' not in {sorted(BACKGROUNDS)}")
    mapping = _parse_mapping(raw.get("mapping"), targets, context)
    return SourceSpec(
        name=name,
        fetcher=_parse_fetcher(raw.get("fetcher"), context),
        license=_str(raw, "license", context),
        attribution=_str(raw, "attribution", context),
        annotation_type=annotation_type,
        background=background,
        mapping=mapping,
        drops=_parse_drops(raw.get("drops"), mapping, context),
        cap=_parse_cap(raw.get("cap"), targets, context),
    )


def load_registry(path: str | Path, target_classes: Collection[str]) -> dict[str, SourceSpec]:
    """Load + validate the registry; returns sources keyed by name, in file order."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise _fail(str(path), "top level must be a mapping with a 'sources' list")
    _check_keys(raw, _TOP_KEYS, str(path))
    entries = raw.get("sources")
    if not isinstance(entries, list) or not entries:
        raise _fail(str(path), "'sources' must be a non-empty list")
    registry: dict[str, SourceSpec] = {}
    for entry in entries:
        spec = parse_source(entry, target_classes)
        if spec.name in registry:
            raise _fail(str(path), f"duplicate source name '{spec.name}'")
        registry[spec.name] = spec
    return registry
