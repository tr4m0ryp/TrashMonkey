"""Shared in-memory Item factory for balance/split tests (no filesystem I/O)."""

from __future__ import annotations

from pathlib import Path

from yolo_waste_sorter.data.dedup import Item


def make_items(class_name: str, source: str, n: int) -> list[Item]:
    return [
        Item(
            key=f"{class_name}/{source}__{i:04d}.png",
            class_name=class_name,
            source=source,
            image=Path(f"/fake/{class_name}/{source}__{i:04d}.png"),
            label=None,
        )
        for i in range(n)
    ]
