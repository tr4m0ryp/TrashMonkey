"""Runtime entrypoint: ``python -m yolo_waste_sorter.deploy``.

Wires the exported model, the thresholds.yaml artifact, and the grab-latest
camera readers from config, then loops forever emitting one JSON decision
line per object. Ctrl-C stops the readers cleanly.
"""

from __future__ import annotations

from pathlib import Path

from yolo_waste_sorter.deploy.runtime import build_runtime
from yolo_waste_sorter.utils.config import load_config


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m yolo_waste_sorter.deploy",
        description="Jetson runtime: 3x MJPEG -> round-robin engine inference -> "
        "T9 consensus -> JSON decision lines (T8).",
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="experiment yaml (default: configs/config.yaml)"
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=None,
        help="thresholds.yaml artifact (default: beside the engine from deploy.engine)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    runtime = build_runtime(cfg, thresholds_path=args.thresholds)
    try:
        runtime.run()
    except KeyboardInterrupt:
        pass
    finally:
        runtime.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
