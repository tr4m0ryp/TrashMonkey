"""Typed Config -> YOLO().train() keyword arguments (the T7 + T5 table).

Every native argument is config-driven; nothing is defaulted here. ``cutmix``
and ``copy_paste`` are both valid train args on every ultralytics release that
satisfies the package's augmentations-hook floor (8.3.226+; ``cutmix`` did not
exist at v8.3.0 but predates 8.3.226), so both pass through.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trashmonkey.utils.config import Config

# Contract-pinned smoke overrides: full cycle on CPU in seconds.
SMOKE_OVERRIDES: dict[str, Any] = {
    "epochs": 1,
    "batch": 2,
    "imgsz": 160,
    "device": "cpu",
    "workers": 0,
}


def build_train_kwargs(cfg: Config, data_yaml: Path) -> dict[str, Any]:
    """Translate the typed config into the exact T7/T5 train() mapping."""
    train, augment = cfg.train, cfg.augment
    return {
        "data": str(data_yaml),
        # T7 fine-tuning recipe (pinned AdamW, full fine-tune, disk cache).
        "epochs": train.epochs,
        "optimizer": train.optimizer,
        "lr0": train.lr0,
        "lrf": train.lrf,
        "momentum": train.momentum,
        "weight_decay": train.weight_decay,
        "warmup_epochs": train.warmup_epochs,
        "batch": train.batch,
        "imgsz": train.imgsz,
        "patience": train.patience,
        "close_mosaic": train.close_mosaic,
        "cache": train.cache,
        "amp": train.amp,
        "deterministic": train.deterministic,
        "seed": cfg.seed,
        "workers": train.workers,
        "freeze": train.freeze,
        # T5 native Ultralytics augmentation args.
        "degrees": augment.degrees,
        "flipud": augment.flipud,
        "fliplr": augment.fliplr,
        "hsv_h": augment.hsv_h,
        "hsv_s": augment.hsv_s,
        "hsv_v": augment.hsv_v,
        "translate": augment.translate,
        "scale": augment.scale,
        "mosaic": augment.mosaic,
        "mixup": augment.mixup,
        "cutmix": augment.cutmix,
        "shear": augment.shear,
        "perspective": augment.perspective,
        "copy_paste": augment.copy_paste,
    }
