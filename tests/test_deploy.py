"""Export tests: format validation and the TensorRT host refusal.

No network, no ultralytics -- every precondition fires before the lazy
ultralytics import.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trashmonkey.deploy import (
    SUPPORTED_FORMATS,
    ExportError,
    ensure_jetson_arch,
    export_engine,
    export_model,
)


def test_export_rejects_unsupported_format(tmp_path: Path) -> None:
    with pytest.raises(ExportError, match="unsupported format 'gguf'"):
        export_model(tmp_path / "best.pt", fmt="gguf", imgsz=640)
    assert "onnx" in SUPPORTED_FORMATS and "engine" in SUPPORTED_FORMATS


def test_engine_export_refuses_non_aarch64_host(tmp_path: Path) -> None:
    with pytest.raises(ExportError, match="refusing to export"):
        ensure_jetson_arch(machine=lambda: "x86_64")
    with pytest.raises(ExportError, match="refusing to export"):
        export_engine(tmp_path / "best.pt", imgsz=640, machine=lambda: "arm64")
    ensure_jetson_arch(machine=lambda: "aarch64")  # deployment device passes
    ensure_jetson_arch(force_host=True, machine=lambda: "x86_64")  # explicit override
    # weights check fires before the lazy ultralytics import (not installed here)
    with pytest.raises(ExportError, match="weights not found"):
        export_engine(tmp_path / "best.pt", imgsz=640, machine=lambda: "aarch64")


def test_non_engine_formats_skip_the_host_guard(tmp_path: Path) -> None:
    # ONNX export has no architecture precondition; the missing-weights check
    # is the first thing to fire even on a non-aarch64 host.
    with pytest.raises(ExportError, match="weights not found"):
        export_model(tmp_path / "best.pt", fmt="onnx", imgsz=640, machine=lambda: "x86_64")
