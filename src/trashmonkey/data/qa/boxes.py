"""YOLO txt label parsing and box geometry (normalized cxcywh)."""

from pathlib import Path
from typing import NamedTuple


class Box(NamedTuple):
    """One YOLO label line: class id + normalized center-format box."""

    class_id: int
    cx: float
    cy: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def aspect(self) -> float:
        """Width/height ratio. Undefined (raises) for zero-height boxes."""
        if self.h <= 0.0:
            raise ZeroDivisionError("aspect undefined for zero-height box")
        return self.w / self.h

    def to_xyxy(self) -> tuple[float, float, float, float]:
        half_w, half_h = self.w / 2.0, self.h / 2.0
        return (self.cx - half_w, self.cy - half_h, self.cx + half_w, self.cy + half_h)

    def edges_touched(self, eps: float) -> int:
        """Count image borders (of 4) this box reaches within eps."""
        x1, y1, x2, y2 = self.to_xyxy()
        return sum((x1 <= eps, y1 <= eps, x2 >= 1.0 - eps, y2 >= 1.0 - eps))


def parse_label_file(path: Path) -> list[Box]:
    """Parse one YOLO txt file. Fails fast on malformed or out-of-range lines."""
    boxes: list[Box] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{lineno}: expected 5 fields, got {len(parts)}: {line!r}")
        try:
            class_id = int(parts[0])
            cx, cy, w, h = (float(p) for p in parts[1:])
        except ValueError as exc:
            raise ValueError(f"{path}:{lineno}: unparseable label line: {line!r}") from exc
        for name, value in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{path}:{lineno}: {name}={value} outside [0, 1]")
        if class_id < 0:
            raise ValueError(f"{path}:{lineno}: negative class id {class_id}")
        boxes.append(Box(class_id, cx, cy, w, h))
    return boxes


def iou_cxcywh(a: Box, b: Box) -> float:
    """IoU of two normalized center-format boxes. Class-agnostic."""
    ax1, ay1, ax2, ay2 = a.to_xyxy()
    bx1, by1, bx2, by2 = b.to_xyxy()
    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0.0 or inter_h <= 0.0:
        return 0.0
    inter = inter_w * inter_h
    union = a.area + b.area - inter
    if union <= 0.0:
        return 0.0
    return inter / union
