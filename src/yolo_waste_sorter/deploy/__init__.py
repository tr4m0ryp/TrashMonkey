"""Deployment-artifact tooling: model export + thresholds.yaml reader.

This repo stops at the artifacts: exported weights (any supported
Ultralytics format -- ONNX by default, TensorRT for NVIDIA edge devices)
and the tuned ``thresholds.yaml``. Wiring those artifacts to cameras and
control hardware is a downstream integration concern and lives outside
this repository. ``load_threshold_params`` is the fail-fast reader an
integration can use to consume the thresholds artifact; the consensus
decision rule itself lives in ``yolo_waste_sorter.models.thresholds``.
"""

from yolo_waste_sorter.deploy.artifacts import load_threshold_params
from yolo_waste_sorter.deploy.export import (
    SUPPORTED_FORMATS,
    ExportError,
    ensure_jetson_arch,
    export_engine,
    export_model,
)

__all__ = [
    "SUPPORTED_FORMATS",
    "ExportError",
    "ensure_jetson_arch",
    "export_engine",
    "export_model",
    "load_threshold_params",
]
