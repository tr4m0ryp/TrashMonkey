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
    of ``yolo_waste_sorter.utils.degrade.build_train_stack``.
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
class EvalConfig:
    """T6 three-tier eval; null fields are filled after the dataset census."""

    val_fraction: float | None
    leave_out_source: str | None
    test2_severities: tuple[int, ...]


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
class DeployConfig:
    """T8 Jetson runtime: camera streams, object window, engine artifact.

    ``cameras`` are the ESP32-CAM MJPEG stream URLs (one grab-latest reader
    thread each); ``window_seconds`` is the per-object vote window (T9);
    ``engine`` is the on-device TensorRT artifact (R5: exported ON the
    Jetson); ``stale_after_s`` drops frames older than the freshness budget;
    ``reconnect_backoff_s`` paces stream reconnects (F11: the cameras, not
    the Orin, are where latency risk lives).
    """

    cameras: tuple[str, ...]
    window_seconds: float
    engine: Path
    reconnect_backoff_s: float
    stale_after_s: float


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
    deploy: DeployConfig
