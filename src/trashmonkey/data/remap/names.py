"""Class-index discovery for detection sources.

Precedence: explicit names from the caller/registry > `names:` in a data.yaml
in the raw tree > classes.txt lines. Anything else fails with a clear message:
rewriting YOLO class ids with a guessed index order would corrupt every label.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from trashmonkey.data.remap.errors import ClassNamesError


def _names_from_data_yaml(path: Path) -> tuple[str, ...]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or "names" not in raw:
        raise ClassNamesError(f"{path}: expected a mapping with a 'names' key")
    names = raw["names"]
    if isinstance(names, list):
        if not names or not all(isinstance(n, str) for n in names):
            raise ClassNamesError(f"{path}: 'names' must be a non-empty list of strings")
        return tuple(names)
    if isinstance(names, dict):
        try:
            items = sorted((int(k), str(v)) for k, v in names.items())
        except (TypeError, ValueError) as exc:
            raise ClassNamesError(f"{path}: 'names' dict keys must be ints, got {names!r}") from exc
        if [k for k, _ in items] != list(range(len(items))):
            raise ClassNamesError(f"{path}: 'names' dict keys must be contiguous from 0")
        return tuple(v for _, v in items)
    raise ClassNamesError(f"{path}: unsupported 'names' type {type(names).__name__}")


def _names_from_classes_txt(path: Path) -> tuple[str, ...]:
    names = tuple(line.strip() for line in path.read_text().splitlines() if line.strip())
    if not names:
        raise ClassNamesError(f"{path}: classes.txt is empty")
    return names


def _unique(candidates: list[tuple[tuple[str, ...], Path]], source: str) -> tuple[str, ...]:
    if len({names for names, _ in candidates}) > 1:
        files = ", ".join(str(p) for _, p in candidates)
        raise ClassNamesError(f"source '{source}': conflicting class-name definitions in {files}")
    return candidates[0][0]


def resolve_class_names(source_root: Path, source: str) -> tuple[str, ...]:
    """Source class-index order: data.yaml `names`, else classes.txt, else fail."""
    yamls = sorted(source_root.rglob("data.yaml")) + sorted(source_root.rglob("data.yml"))
    if yamls:
        return _unique([(_names_from_data_yaml(p), p) for p in yamls], source)
    txts = sorted(source_root.rglob("classes.txt"))
    if txts:
        return _unique([(_names_from_classes_txt(p), p) for p in txts], source)
    raise ClassNamesError(
        f"source '{source}': cannot determine the class-index order -- no explicit names "
        f"were provided and no data.yaml/classes.txt exists under {source_root}"
    )
