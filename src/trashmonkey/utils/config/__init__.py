"""Typed, fail-fast config access: ``load_config()`` -> frozen ``Config``."""

from trashmonkey.utils.config.loader import (
    DEFAULT_CONFIG_PATH,
    ConfigError,
    load_config,
)
from trashmonkey.utils.config.schema import (
    AugmentConfig,
    Config,
    DefocusConfig,
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
