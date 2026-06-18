"""Frozen dataclass schema mirroring configs/config.yaml exactly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathsConfig:
    raw: Path
    interim: Path
    processed: Path
    external: Path
    models: Path
    reports: Path


@dataclass(frozen=True)
class ExperimentConfig:
    name: str


@dataclass(frozen=True)
class ModelConfig:
    base: str
    imgsz: int


@dataclass(frozen=True)
class TrainConfig:
    """T7 recipe: pinned AdamW, full fine-tune (freeze None), disk cache."""

    epochs: int
    optimizer: str
    lr0: float
    lrf: float
    momentum: float
    weight_decay: float
    warmup_epochs: float
    batch: int
    imgsz: int
    patience: int
    close_mosaic: int
    cache: str
    amp: bool
    deterministic: bool
    freeze: int | None
    workers: int
    # Cosine LR schedule toggle. False = the current linear decay to lr0*lrf.
    # Exposed as a knob so a seed-fixed A/B can evaluate it before adoption;
    # keep False until that A/B confirms a win (the recipe is otherwise pinned).
    cos_lr: bool
    # Ultralytics native inverse-frequency class weighting (its `cls_pw` train
    # arg), range [0, 1]; 0.0 = OFF (current pinned behaviour). A seed-fixed A/B
    # can raise it to up-weight minority classes (e.g. organic) before adoption.
    cls_pw: float


@dataclass(frozen=True)
class ProbConfig:
    p: float


@dataclass(frozen=True)
class ImageCompressionConfig:
    quality_range: tuple[int, int]
    p: float


@dataclass(frozen=True)
class MotionBlurConfig:
    blur_limit: tuple[int, int]
    p: float


@dataclass(frozen=True)
class DefocusConfig:
    radius: tuple[int, int]
    p: float


@dataclass(frozen=True)
class DownscaleConfig:
    scale_range: tuple[float, float]
    p: float


@dataclass(frozen=True)
class Esp32StackConfig:
    """T5 Albumentations degradation stack simulating the ESP32-CAM OV2640.

    Field names are the snake_case Albumentations transform names and the
    inner keys are the transforms' constructor kwargs -- the exact contract
    of ``trashmonkey.utils.degrade.build_train_stack``.
    """

    image_compression: ImageCompressionConfig
    iso_noise: ProbConfig
    gauss_noise: ProbConfig
    motion_blur: MotionBlurConfig
    defocus: DefocusConfig
    planckian_jitter: ProbConfig
    downscale: DownscaleConfig
    random_brightness_contrast: ProbConfig


@dataclass(frozen=True)
class AugmentConfig:
    """T5 native Ultralytics augmentation args plus the ESP32 stack."""

    degrees: float
    flipud: float
    fliplr: float
    hsv_h: float
    hsv_s: float
    hsv_v: float
    translate: float
    scale: float
    mosaic: float
    mixup: float
    cutmix: float
    shear: float
    perspective: float
    copy_paste: float
    esp32_stack: Esp32StackConfig


@dataclass(frozen=True)
class CleanHoldoutConfig:
    """Plain-background clean-presentation holdout carved from training sources.

    ``fraction`` of each named source is held out (seeded, deterministic) as a
    deployment-distribution test set distinct from the T6 leave-one-source-out
    TEST-1. Consumed by a downstream task; unused until wired in.
    """

    fraction: float
    sources: tuple[str, ...]


@dataclass(frozen=True)
class EscalationConfig:
    """Per-metric floors gating the model-size escalation decision (yolo11n->s).

    Replaces the old hard 0.95 floor: an escalation is triggered only when the
    overall mAP50 or any per-class mAP50/recall falls below these thresholds.
    """

    overall_map50: float
    class_map50: float
    class_recall: float


@dataclass(frozen=True)
class LabelFilterConfig:
    """Quality gate dropping degenerate auto-boxes before they enter training.

    ``drop_methods`` names boxing methods whose outputs are discarded outright
    (e.g. the centerbox fallback). The fractional bounds are box-area fractions
    of the image. Consumed by a downstream task; defaults keep current behaviour.
    """

    min_confidence: float
    max_box_frac: float
    min_box_frac: float
    drop_methods: tuple[str, ...]


@dataclass(frozen=True)
class EvalConfig:
    """T6 three-tier eval; null fields are filled after the dataset census."""

    val_fraction: float | None
    leave_out_source: str | None
    test2_severities: tuple[int, ...]
    clean_holdout: CleanHoldoutConfig
    escalation: EscalationConfig
    label_filter: LabelFilterConfig


@dataclass(frozen=True)
class SweepConfig:
    tau_frame: tuple[float, ...]
    min_votes: tuple[int, ...]
    high_water: tuple[float, ...]


@dataclass(frozen=True)
class ThresholdsConfig:
    """T9 rest-bin consensus rule starting values plus the sweep grids."""

    conf_floor: float
    tau_frame: float
    min_votes: int
    high_water: float
    sweep: SweepConfig


@dataclass(frozen=True)
class Config:
    seed: int
    paths: PathsConfig
    experiment: ExperimentConfig
    model: ModelConfig
    classes: tuple[str, ...]
    train: TrainConfig
    augment: AugmentConfig
    eval: EvalConfig
    thresholds: ThresholdsConfig
