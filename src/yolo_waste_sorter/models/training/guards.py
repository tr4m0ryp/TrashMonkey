"""Hard guards on the T7 recipe -- fail before any GPU time is spent.

Each guard encodes a verified ultralytics 8.3.x failure mode from the research
doc (F1, F4) or a reproducibility rule (no auto batch). A config carrying a
forbidden value raises ``ValueError`` with the evidence inline.
"""

from __future__ import annotations

from yolo_waste_sorter.utils.config import Config

# First ultralytics release whose train() accepts the validated `augmentations`
# kwarg: cfg.check_dict_alignment(allowed_custom_keys={"augmentations"}) and
# data.augment.Albumentations(p=1.0, transforms=getattr(hyp, "augmentations",
# None)) both land in v8.3.226 (verified against the GitHub tags; absent in
# v8.3.225 and earlier, present through v8.3.253 and 8.4.x).
MIN_ULTRALYTICS = (8, 3, 226)


def validate_train_config(cfg: Config) -> None:
    """Reject configs that would silently change or break the T7 recipe."""
    optimizer = cfg.train.optimizer
    if isinstance(optimizer, str) and optimizer.lower() == "auto":
        raise ValueError(
            "train.optimizer='auto' is forbidden (research F1): ultralytics picks "
            "SGD over AdamW once iterations exceed 10k, so growing the dataset "
            "silently changes the recipe; pin optimizer='AdamW'"
        )
    cache: object = cfg.train.cache
    if cache is True or (isinstance(cache, str) and cache.lower() == "ram"):
        raise ValueError(
            "train.cache='ram' (or True) is forbidden (research F4): RAM caching's "
            "ThreadPool fill order breaks training determinism even with a fixed "
            "seed; use cache='disk'"
        )
    batch: object = cfg.train.batch
    if not isinstance(batch, int) or isinstance(batch, bool) or batch <= 0:
        raise ValueError(
            f"train.batch must be a positive int, got {batch!r}: batch=-1 "
            "auto-sizing is hardware-dependent and not reproducible (T7)"
        )


def require_augmentations_support(version: str) -> None:
    """Fail fast if the installed ultralytics predates the augmentations hook."""
    try:
        parsed = tuple(int(part) for part in version.split(".")[:3])
    except ValueError as exc:
        raise RuntimeError(f"cannot parse ultralytics version {version!r}") from exc
    if parsed < MIN_ULTRALYTICS:
        floor = ".".join(str(part) for part in MIN_ULTRALYTICS)
        raise RuntimeError(
            f"ultralytics {version} does not accept train(augmentations=[...]): the "
            f"validated custom-Albumentations hook was added in {floor} "
            "(cfg/__init__.py allowed_custom_keys and Albumentations(transforms=...)). "
            f"Install ultralytics>={floor},<9 so the T5 degradation stack reaches the "
            "training pipeline."
        )
