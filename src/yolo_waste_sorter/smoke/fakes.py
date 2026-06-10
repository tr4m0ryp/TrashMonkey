"""FAKE_MODEL=1 path: a sys.modules-injected ultralytics double (task 016).

Mirrors the exact DetMetrics / predict surfaces the chain consumes -- the same
shape the offline test suites mock: ``train()`` writes a plausible best.pt and
returns metrics; ``val()`` adds the Confidence/Precision-Recall-F1
``curves_results``; ``predict()`` emits one detection per image, classed from
the emitted filename's ``<class>__`` prefix (val frames) or a low-confidence
class-0 hit for wilderness frames, so the threshold tuner sees plausible
known-object votes AND wilderness probes.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

FAKE_VERSION = "8.3.253"  # satisfies the >=8.3.226 augmentations-hook floor
KNOWN_SCORE = 0.92  # above the smoke sweep taus -> objects sort to their bin
WILDERNESS_SCORE = 0.30  # below the smoke sweep taus -> wilderness rests


def _fake_metrics(classes: tuple[str, ...]) -> SimpleNamespace:
    """DetMetrics-like object: per-class arrays, results_dict, curve entries."""
    n = len(classes)
    points = np.linspace(0.0, 1.0, 21)
    p_curve = np.tile(np.clip(0.6 + 0.75 * points, 0.0, 1.0), (n, 1))
    r_curve = np.tile(np.clip(1.0 - points, 0.0, 1.0), (n, 1))
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + 1e-9)
    box = SimpleNamespace(
        ap_class_index=list(range(n)),
        p=[0.94] * n,
        r=[0.95] * n,
        ap50=[0.96] * n,
        ap=[0.80] * n,
        map50=0.97,
        map=0.80,
    )
    return SimpleNamespace(
        box=box,
        names=dict(enumerate(classes)),
        results_dict={"metrics/mAP50(B)": 0.97, "fitness": 0.85},
        curves_results=[
            [points, f1_curve, "Confidence", "F1"],
            [points, p_curve, "Confidence", "Precision"],
            [points, r_curve, "Confidence", "Recall"],
        ],
    )


def _detection_for(stem: str, class_ids: dict[str, int]) -> tuple[int, float]:
    """(class_id, score) for one emitted image, keyed by its flattened stem."""
    prefix = stem.split("__", 1)[0]
    class_id = class_ids.get(prefix)
    if class_id is not None:
        return class_id, KNOWN_SCORE  # val frame: vote for the true class
    return 0, WILDERNESS_SCORE  # wilderness frame: weak, unqualified vote


def build_fake_ultralytics(classes: tuple[str, ...]) -> types.ModuleType:
    """A module exposing __version__ and YOLO with train/val/predict doubles."""
    class_ids = {name: class_id for class_id, name in enumerate(classes)}

    class FakeYOLO:
        def __init__(self, model: str) -> None:
            self.model = model
            self.trainer: SimpleNamespace | None = None

        def train(self, **kwargs: Any) -> SimpleNamespace:
            run_dir = Path(str(kwargs["project"])) / str(kwargs["name"])
            (run_dir / "weights").mkdir(parents=True, exist_ok=True)
            best = run_dir / "weights" / "best.pt"
            best.write_bytes(b"fake smoke checkpoint")
            self.trainer = SimpleNamespace(
                best=str(best),
                save_dir=str(run_dir),
                metrics={"metrics/mAP50(B)": 0.97, "fitness": 0.85},
            )
            return _fake_metrics(classes)

        def val(self, **kwargs: Any) -> SimpleNamespace:
            return _fake_metrics(classes)

        def predict(self, source: list[str], **kwargs: Any) -> Iterator[SimpleNamespace]:
            for path in source:
                class_id, score = _detection_for(Path(path).stem, class_ids)
                yield SimpleNamespace(
                    path=path,
                    boxes=SimpleNamespace(cls=[float(class_id)], conf=[score]),
                )

    module = types.ModuleType("ultralytics")
    setattr(module, "__version__", FAKE_VERSION)
    setattr(module, "YOLO", FakeYOLO)
    return module


def install_fake_ultralytics(classes: tuple[str, ...]) -> None:
    """Make every later ``import ultralytics`` resolve to the fake module."""
    sys.modules["ultralytics"] = build_fake_ultralytics(classes)
