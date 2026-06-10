"""Remap stage (T1): source labels -> the six target classes, DROPs -> wilderness.

Copies data/raw/<source>/ into data/interim/remapped/<class>/ (det label txts
rewritten to target ids) and data/interim/wilderness/ for DROPs; writes one
RemapManifest YAML per source. Public surface re-exported here.
"""

from yolo_waste_sorter.data.remap.errors import ClassNamesError, RemapError, UnmappedLabelError
from yolo_waste_sorter.data.remap.layout import IMAGE_SUFFIXES
from yolo_waste_sorter.data.remap.manifest import (
    MANIFESTS_DIRNAME,
    REMAPPED_DIRNAME,
    WILDERNESS_DIRNAME,
    RemapManifest,
    manifest_path,
)
from yolo_waste_sorter.data.remap.names import resolve_class_names
from yolo_waste_sorter.data.remap.pipeline import remap_source, remap_sources

__all__ = [
    "IMAGE_SUFFIXES",
    "MANIFESTS_DIRNAME",
    "REMAPPED_DIRNAME",
    "WILDERNESS_DIRNAME",
    "ClassNamesError",
    "RemapError",
    "RemapManifest",
    "UnmappedLabelError",
    "manifest_path",
    "remap_source",
    "remap_sources",
    "resolve_class_names",
]
