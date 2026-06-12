"""Deployment package: grab-latest streams, consensus runtime, export, checks.

Deployment is target-agnostic: weights export to any supported Ultralytics
format (ONNX by default, TensorRT for NVIDIA edge devices such as the Jetson
Orin Nano), and the runtime loads whatever artifact ``deploy.model`` points
at. The runtime IMPORTS the shared consensus rule from
``yolo_waste_sorter.models.thresholds`` -- it is never reimplemented here.
Emitted decisions use the literal string ``"rest"`` (``REST_LABEL``) as the
wire form of the REST sentinel; "rest" is NOT a trained class.
"""

from yolo_waste_sorter.deploy.artifacts import load_threshold_params
from yolo_waste_sorter.deploy.check_env import CheckResult, format_table, run_checks
from yolo_waste_sorter.deploy.export import (
    SUPPORTED_FORMATS,
    ExportError,
    ensure_jetson_arch,
    export_engine,
    export_model,
)
from yolo_waste_sorter.deploy.runtime import (
    REST_LABEL,
    DecisionEvent,
    DeployError,
    EmitFn,
    PredictFn,
    Runtime,
    build_runtime,
    load_model_predictor,
)
from yolo_waste_sorter.deploy.streams import CameraReader, Frame, StreamError, start_readers

__all__ = [
    "REST_LABEL",
    "SUPPORTED_FORMATS",
    "CameraReader",
    "CheckResult",
    "DecisionEvent",
    "DeployError",
    "EmitFn",
    "ExportError",
    "Frame",
    "PredictFn",
    "Runtime",
    "StreamError",
    "build_runtime",
    "ensure_jetson_arch",
    "export_engine",
    "export_model",
    "format_table",
    "load_model_predictor",
    "load_threshold_params",
    "run_checks",
    "start_readers",
]
