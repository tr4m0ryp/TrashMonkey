"""Typed, fail-fast config access: ``load_config()`` -> frozen ``Config``."""

from yolo_waste_sorter.utils.config.loader import (
    DEFAULT_CONFIG_PATH,
    ConfigError,
    load_config,
)
from yolo_waste_sorter.utils.config.schema import (
    AugmentConfig,
    Config,
    DefocusConfig,
    DeployConfig,
    DownscaleConfig,
    Esp32StackConfig,
    EvalConfig,
    ExperimentConfig,
    ImageCompressionConfig,
    ModelConfig,
    MotionBlurConfig,
    PathsConfig,
    ProbConfig,
    SweepConfig,
    ThresholdsConfig,
    TrainConfig,
)

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "AugmentConfig",
    "Config",
    "ConfigError",
    "DefocusConfig",
    "DeployConfig",
    "DownscaleConfig",
    "Esp32StackConfig",
    "EvalConfig",
    "ExperimentConfig",
    "ImageCompressionConfig",
    "ModelConfig",
    "MotionBlurConfig",
    "PathsConfig",
    "ProbConfig",
    "SweepConfig",
    "ThresholdsConfig",
    "TrainConfig",
    "load_config",
]
