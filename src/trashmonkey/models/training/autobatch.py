"""Hardware-adaptive, reproducible batch/worker sizing for the T7 fine-tune.

ultralytics' built-in ``batch=-1`` autobatch probes free VRAM at runtime, so
the chosen batch drifts run-to-run and across cards -- the training guards
forbid it (reproducibility, T7). This module replaces it with a *static*
VRAM -> (batch, workers) table: the resolved integers are a pure function of
the detected card, so a given runtime always yields the same recipe and the
value lands in ``runs.jsonl`` like any other pinned arg.

``batch`` is capped at 64 = ultralytics' nominal batch size (``nbs``). At
batch <= 64 the trainer keeps gradient accumulation at 1 and its weight-decay /
LR scaling is identical to the batch-16 recipe, so a bigger GPU buys throughput
(fewer, larger steps) without changing the optimisation. For a nano model the
data loader -- not the GPU -- is the bottleneck, so ``workers`` scales with the
tier too, clamped to the host core count.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass

from trashmonkey.utils.config import Config

# (min_total_vram_gb, batch, workers). Ordered high -> low; first match wins.
# The 0.0 tier is the pinned configs/config.yaml default (small GPU / CPU).
# Thresholds sit BELOW each card's nominal RAM because torch reports usable GiB,
# which is a few percent under the marketing GB: an A100-"40GB" reports ~39.5
# GiB, L4-"24GB" ~23.6, T4-"16GB" ~15.6. The top cut is 38 (not 40) so the
# A100-40 lands on batch 64 instead of slipping into the 48 tier.
_TIERS: tuple[tuple[float, int, int], ...] = (
    (38.0, 64, 16),  # A100-40 (~39.5 GiB) / A100-80 / H100-80
    (20.0, 48, 12),  # L4 24GB (~23.6 GiB)
    (14.0, 32, 8),   # T4 / G4 16GB (~15.6 GiB)
    (0.0, 16, 8),    # small GPU / CPU -> config.yaml default
)


@dataclass(frozen=True)
class RuntimeProfile:
    """The accelerator the run landed on and the recipe knobs it implies."""

    device: str  # "cuda:0" or "cpu"
    name: str  # GPU model string, or "cpu"
    total_memory_gb: float
    batch: int
    workers: int

    def summary(self) -> str:
        head = "cpu" if self.device == "cpu" else f"{self.name} ({self.total_memory_gb:.0f} GB)"
        return f"{head} -> batch={self.batch}, workers={self.workers}"


def _select(total_memory_gb: float, cpu_count: int) -> tuple[int, int]:
    for floor, batch, workers in _TIERS:
        if total_memory_gb >= floor:
            return batch, min(workers, max(1, cpu_count))
    raise AssertionError("the 0.0 tier always matches")  # unreachable; defensive


def detect_runtime() -> RuntimeProfile:
    """Probe the active CUDA device. torch is optional -> CPU fallback."""
    cpu_count = os.cpu_count() or 1
    try:
        import torch
    except ImportError:
        batch, workers = _select(0.0, cpu_count)
        return RuntimeProfile("cpu", "cpu", 0.0, batch, workers)
    if not torch.cuda.is_available():
        batch, workers = _select(0.0, cpu_count)
        return RuntimeProfile("cpu", "cpu", 0.0, batch, workers)
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024**3)
    batch, workers = _select(total_gb, cpu_count)
    return RuntimeProfile("cuda:0", str(props.name), total_gb, batch, workers)


def apply_runtime(
    cfg: Config, profile: RuntimeProfile | None = None
) -> tuple[Config, RuntimeProfile]:
    """Return ``cfg`` with batch/workers resolved for the detected accelerator.

    ``cfg`` is frozen, so a new ``Config`` is returned. Pass an explicit
    ``profile`` to reuse one detection across notebook cells (or in tests).
    """
    if profile is None:
        profile = detect_runtime()
    resolved_train = dataclasses.replace(
        cfg.train, batch=profile.batch, workers=profile.workers
    )
    return dataclasses.replace(cfg, train=resolved_train), profile
