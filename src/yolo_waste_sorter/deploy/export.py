"""On-device TensorRT export + engine smoke test (T8).

Runs ON the Jetson: TensorRT engines are bound to the TRT version and the
device's compute capability (R5), so this script REFUSES to run on a
non-aarch64 host unless ``--force-host`` is passed (e.g. an aarch64-adjacent
CI experiment -- the produced engine still will not deserialize on the Orin).
Export is FP16, static batch 1, at the training imgsz; a dummy-image
inference on the produced engine must succeed before the artifact is trusted.

CLI: ``python -m yolo_waste_sorter.deploy.export --weights models/best.pt``
"""

from __future__ import annotations

import platform
from collections.abc import Callable
from pathlib import Path

import numpy as np


class ExportError(Exception):
    """Export preconditions or the engine smoke test failed."""


def ensure_jetson_arch(
    *, force_host: bool = False, machine: Callable[[], str] = platform.machine
) -> None:
    """Fail unless we are on aarch64 (engines are device-bound, R5)."""
    arch = machine()
    if arch == "aarch64" or force_host:
        return
    raise ExportError(
        f"refusing to export a TensorRT engine on {arch!r}: engines do not "
        "deserialize across TRT versions / compute capabilities (R5) -- run "
        "this on the Jetson, or pass --force-host if you really mean it"
    )


def export_engine(
    weights: Path,
    *,
    imgsz: int,
    force_host: bool = False,
    machine: Callable[[], str] = platform.machine,
) -> Path:
    """Export ``weights`` -> FP16 TensorRT engine and smoke-test it.

    Returns the engine path. ultralytics is imported lazily -- it only
    exists in the Jetson runtime image, never in the dev/test environment.
    """
    ensure_jetson_arch(force_host=force_host, machine=machine)
    if not weights.is_file():
        raise ExportError(f"weights not found: {weights}")
    from ultralytics import YOLO

    exported = YOLO(str(weights)).export(format="engine", half=True, imgsz=imgsz, batch=1)
    engine = Path(str(exported))
    if not engine.is_file():
        raise ExportError(f"export reported {engine} but the file does not exist")
    smoke_test_engine(engine, imgsz=imgsz)
    return engine


def smoke_test_engine(engine: Path, *, imgsz: int) -> None:
    """One dummy-image inference on the produced engine; raise on any failure.

    The dummy frame is white -- the deployment background -- so zero
    detections are expected and fine; only a crashing engine fails here.
    """
    from ultralytics import YOLO

    dummy = np.full((imgsz, imgsz, 3), 255, dtype=np.uint8)
    try:
        YOLO(str(engine), task="detect").predict(dummy, verbose=False)
    except Exception as err:  # noqa: BLE001 -- engine load/infer can fail many ways
        raise ExportError(f"engine smoke inference failed for {engine}: {err}") from err


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m yolo_waste_sorter.deploy.export",
        description="On-Jetson TensorRT FP16 export (batch 1) + engine smoke test (T8/R5).",
    )
    parser.add_argument("--weights", type=Path, required=True, help="trained best.pt")
    parser.add_argument(
        "--config", type=Path, default=None, help="experiment yaml (default: configs/config.yaml)"
    )
    parser.add_argument(
        "--force-host",
        action="store_true",
        help="export on a non-aarch64 host anyway (the engine will NOT run on the Orin)",
    )
    args = parser.parse_args(argv)

    from yolo_waste_sorter.utils.config import load_config

    cfg = load_config(args.config)
    try:
        engine = export_engine(args.weights, imgsz=cfg.train.imgsz, force_host=args.force_host)
    except ExportError as err:
        print(f"export failed: {err}", file=sys.stderr)
        return 2
    print(f"engine:  {engine}")
    print(f"imgsz:   {cfg.train.imgsz} (FP16, batch 1)")
    print(
        "next:    place thresholds.yaml beside the engine and run "
        "python -m yolo_waste_sorter.deploy"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
