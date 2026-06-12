"""Model export to a deployment format + artifact smoke test.

Exports trained weights to any supported Ultralytics format (ONNX by
default -- portable across runtimes) and smoke-tests the produced artifact
with one dummy-image inference before it is trusted. TensorRT (``engine``)
is the special case: engines are bound to the TensorRT version and the
device's compute capability, so engine export REFUSES to run on a
non-aarch64 host unless ``--force-host`` is passed (the produced engine
still will not deserialize on a different device).

CLI: ``python -m yolo_waste_sorter.deploy.export --weights models/best.pt``
"""

from __future__ import annotations

import platform
from collections.abc import Callable
from pathlib import Path

import numpy as np

# Curated subset of Ultralytics export formats; "engine" (TensorRT) is the
# only one with a host-architecture precondition.
SUPPORTED_FORMATS: tuple[str, ...] = (
    "onnx",
    "engine",
    "torchscript",
    "openvino",
    "coreml",
    "tflite",
    "ncnn",
)


class ExportError(Exception):
    """Export preconditions or the artifact smoke test failed."""


def ensure_jetson_arch(
    *, force_host: bool = False, machine: Callable[[], str] = platform.machine
) -> None:
    """Fail unless we are on aarch64 (TensorRT engines are device-bound)."""
    arch = machine()
    if arch == "aarch64" or force_host:
        return
    raise ExportError(
        f"refusing to export a TensorRT engine on {arch!r}: engines do not "
        "deserialize across TRT versions / compute capabilities -- run this "
        "on the deployment device, or pass --force-host if you really mean it"
    )


def export_model(
    weights: Path,
    *,
    fmt: str = "onnx",
    imgsz: int,
    half: bool | None = None,
    force_host: bool = False,
    machine: Callable[[], str] = platform.machine,
) -> Path:
    """Export ``weights`` to ``fmt`` (static batch 1) and smoke-test the artifact.

    ``half`` defaults to FP16 for TensorRT engines and full precision
    elsewhere (ONNX FP16 needs a GPU exporter). Returns the artifact path.
    ultralytics is imported lazily -- it is absent in the dev/test environment.
    """
    if fmt not in SUPPORTED_FORMATS:
        raise ExportError(f"unsupported format {fmt!r}; choose one of {SUPPORTED_FORMATS}")
    if fmt == "engine":
        ensure_jetson_arch(force_host=force_host, machine=machine)
    if not weights.is_file():
        raise ExportError(f"weights not found: {weights}")
    use_half = (fmt == "engine") if half is None else half
    from ultralytics import YOLO

    exported = YOLO(str(weights)).export(format=fmt, half=use_half, imgsz=imgsz, batch=1)
    artifact = Path(str(exported))
    if not artifact.exists():
        raise ExportError(f"export reported {artifact} but the path does not exist")
    smoke_test_artifact(artifact, imgsz=imgsz)
    return artifact


def export_engine(
    weights: Path,
    *,
    imgsz: int,
    force_host: bool = False,
    machine: Callable[[], str] = platform.machine,
) -> Path:
    """TensorRT FP16 convenience wrapper around :func:`export_model`."""
    return export_model(
        weights, fmt="engine", imgsz=imgsz, half=True, force_host=force_host, machine=machine
    )


def smoke_test_artifact(artifact: Path, *, imgsz: int) -> None:
    """One dummy-image inference on the exported artifact; raise on any failure.

    The dummy frame is plain white -- the assumed deployment background -- so
    zero detections are expected and fine; only a crashing artifact fails here.
    """
    from ultralytics import YOLO

    dummy = np.full((imgsz, imgsz, 3), 255, dtype=np.uint8)
    try:
        YOLO(str(artifact), task="detect").predict(dummy, verbose=False)
    except Exception as err:  # noqa: BLE001 -- artifact load/infer can fail many ways
        raise ExportError(f"smoke inference failed for {artifact}: {err}") from err


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m yolo_waste_sorter.deploy.export",
        description="Export trained weights to a deployment format (batch 1) + smoke test.",
    )
    parser.add_argument("--weights", type=Path, required=True, help="trained best.pt")
    parser.add_argument(
        "--format",
        dest="fmt",
        choices=SUPPORTED_FORMATS,
        default="onnx",
        help="export format (default: onnx; engine = TensorRT, exported on-device)",
    )
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="FP16 export (default: on for engine, off otherwise)",
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="experiment yaml (default: configs/config.yaml)"
    )
    parser.add_argument(
        "--force-host",
        action="store_true",
        help="build a TensorRT engine on a non-aarch64 host anyway (it will NOT "
        "run on the deployment device)",
    )
    args = parser.parse_args(argv)

    from yolo_waste_sorter.utils.config import load_config

    cfg = load_config(args.config)
    try:
        artifact = export_model(
            args.weights,
            fmt=args.fmt,
            imgsz=cfg.train.imgsz,
            half=args.half,
            force_host=args.force_host,
        )
    except ExportError as err:
        print(f"export failed: {err}", file=sys.stderr)
        return 2
    print(f"artifact: {artifact}")
    print(f"format:   {args.fmt} (imgsz {cfg.train.imgsz}, batch 1)")
    print(
        "next:     point deploy.model at the artifact, place thresholds.yaml "
        "beside it, and run python -m yolo_waste_sorter.deploy"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
