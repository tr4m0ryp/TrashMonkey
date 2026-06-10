"""CLI: python -m yolo_waste_sorter.data.download [--source NAME] [--force]"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from yolo_waste_sorter.data.download.errors import DatasetConfigError, DownloadError
from yolo_waste_sorter.data.download.pipeline import download_source
from yolo_waste_sorter.data.download.registry import load_registry
from yolo_waste_sorter.utils.config import load_config


def _target_classes(config_path: Path) -> list[str]:
    classes = load_config(config_path).classes
    if not classes:
        raise DatasetConfigError(f"'classes' in {config_path} must be a non-empty list")
    return list(classes)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch registered sources into data/raw/")
    parser.add_argument("--config", type=Path, default=Path("configs/datasets.yaml"))
    parser.add_argument("--classes-config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--source", action="append", default=None, help="restrict to this source (repeatable)"
    )
    parser.add_argument("--force", action="store_true", help="refetch even if already present")
    args = parser.parse_args(argv)

    try:
        registry = load_registry(args.config, _target_classes(args.classes_config))
        names = args.source or list(registry)
        unknown = [n for n in names if n not in registry]
        if unknown:
            raise DatasetConfigError(f"unknown source(s) {unknown}; registered: {list(registry)}")
        for name in names:
            spec = registry[name]
            result = download_source(spec, args.raw_root, force=args.force)
            print(
                f"{result.source}: {result.action} sha256={result.sha256} "
                f"files={result.file_count} -> {result.dest}"
            )
            if spec.fetcher.sha256 is None and result.action == "fetched":
                print(
                    f"  note: pin sha256: {result.sha256} into {args.config} "
                    f"(entry '{name}') so future runs verify it"
                )
    except DownloadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
