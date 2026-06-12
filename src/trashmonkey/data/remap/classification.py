"""Classification-source planning: class-folder trees, nested splits, case-insensitive.

An image's label is the nearest ancestor folder whose name matches a mapping
key case-insensitively; this handles flat `<class>/`, nested `<split>/<class>/`,
and `<split>/<class>/images/` layouts generically. Images under no matching
folder are unmapped -> hard error (the mapping must be total).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trashmonkey.data.remap.errors import UnmappedLabelError
from trashmonkey.data.remap.layout import is_image

# Folder names that organize files rather than label them; skipped when naming
# the offending folder in an unmapped-label error.
_STRUCTURAL_DIRNAMES = frozenset({"images", "image", "imgs", "img", "labels", "data"})


@dataclass(frozen=True)
class ClsItem:
    """One classification image routed to a target class or DROP."""

    src: Path
    target: str  # target class name, or registry.DROP


def _route(image: Path, source_root: Path, lookup: dict[str, str]) -> str | None:
    for ancestor in image.parents:
        if ancestor == source_root:
            return None
        target = lookup.get(ancestor.name.lower())
        if target is not None:
            return target
    return None


def _unmapped_folder(image: Path, source_root: Path) -> str:
    for ancestor in image.parents:
        if ancestor == source_root:
            return "(source root, no class folder)"
        if ancestor.name.lower() not in _STRUCTURAL_DIRNAMES:
            return ancestor.name
    return "(source root, no class folder)"


def plan_classification(
    source_root: Path, source: str, lookup: dict[str, str]
) -> tuple[list[ClsItem], list[str]]:
    """Route every image under source_root; returns (items, skipped relative paths).

    Raises UnmappedLabelError listing every unmapped class folder before any
    file is copied.
    """
    items: list[ClsItem] = []
    skipped: list[str] = []
    unmapped: set[str] = set()
    for path in sorted(source_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(source_root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if not is_image(path):
            skipped.append(rel.as_posix())
            continue
        target = _route(path, source_root, lookup)
        if target is None:
            unmapped.add(_unmapped_folder(path, source_root))
            continue
        items.append(ClsItem(src=path, target=target))
    if unmapped:
        raise UnmappedLabelError(
            f"source '{source}': unmapped label(s) {sorted(unmapped)} -- every class folder "
            f"must appear in the mapping or in drops (the mapping must be total)"
        )
    return items, skipped
