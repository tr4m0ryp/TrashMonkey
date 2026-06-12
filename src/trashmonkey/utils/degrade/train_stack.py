"""Albumentations training stack simulating the OV2640/ESP32-CAM signature (T5).

Every transform is image-only (bbox-safe): JPEG blocking, AGC/ISO noise,
motion + fixed-focus blur, AWB drift, and low effective resolution -- the
measured deployment defects (research F7), not generic presets.

Parameters come from ``cfg.augment.esp32_stack`` with snake_case keys matching
the transform names. Missing keys fail fast; nothing is silently defaulted.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_ALBUMENTATIONS_INSTALL_HINT = (
    "albumentations is required for the ESP32 training augmentation stack but is "
    "not installed; install it with: pip install 'albumentations>=2.0,<3' "
    "(it is a declared dependency of trashmonkey -- a project install pulls it in)."
)

# (config key under cfg.augment.esp32_stack, albumentations class name).
# Order is the application order of the stack.
_ESP32_STACK: tuple[tuple[str, str], ...] = (
    ("image_compression", "ImageCompression"),
    ("iso_noise", "ISONoise"),
    ("gauss_noise", "GaussNoise"),
    ("motion_blur", "MotionBlur"),
    ("defocus", "Defocus"),
    ("planckian_jitter", "PlanckianJitter"),
    ("downscale", "Downscale"),
    ("random_brightness_contrast", "RandomBrightnessContrast"),
)


def _node_get(node: Any, key: str, path: str) -> Any:
    """Read ``key`` from a config node that may be a mapping or an attribute object."""
    if isinstance(node, Mapping):
        if key in node:
            return node[key]
    elif hasattr(node, key):
        return getattr(node, key)
    raise KeyError(
        f"missing config key '{path}.{key}' -- the ESP32 degradation stack requires it; "
        "add it to configs/*.yaml under augment.esp32_stack"
    )


def _node_items(node: Any, path: str) -> dict[str, Any]:
    """Materialize a config node's key/value pairs from a mapping or attribute object."""
    if isinstance(node, Mapping):
        return {str(key): value for key, value in node.items()}
    if hasattr(node, "__dict__"):
        return {key: value for key, value in vars(node).items() if not key.startswith("_")}
    raise TypeError(
        f"config node '{path}' must be a mapping or an attribute object, "
        f"got {type(node).__name__}"
    )


def _to_kwargs(node: Any, path: str) -> dict[str, Any]:
    """Convert a transform's config node to constructor kwargs (YAML lists -> tuples)."""
    kwargs = {
        key: tuple(value) if isinstance(value, list) else value
        for key, value in _node_items(node, path).items()
    }
    if "p" not in kwargs:
        raise KeyError(
            f"missing config key '{path}.p' -- every ESP32 stack transform must declare "
            "its application probability explicitly"
        )
    return kwargs


def build_train_stack(cfg: Any) -> list[Any]:
    """Build the ESP32-CAM degradation augmentation stack for training (T5).

    Args:
        cfg: Config exposing ``cfg.augment.esp32_stack`` (attribute or mapping
            access), holding one snake_case entry per transform with its
            parameters, e.g. ``image_compression: {quality_range: [50, 85], p: 0.5}``.

    Returns:
        List of instantiated image-only Albumentations transforms, in stack
        order, ready to pass to Ultralytics ``train(augmentations=[...])``.

    Raises:
        ImportError: albumentations is not installed (actionable message).
        KeyError: a required config key is missing (fail fast, no defaults).
        RuntimeError: the installed albumentations lacks a required transform.
    """
    try:
        import albumentations
    except ImportError as exc:
        raise ImportError(_ALBUMENTATIONS_INSTALL_HINT) from exc

    stack_cfg = _node_get(_node_get(cfg, "augment", "cfg"), "esp32_stack", "cfg.augment")
    stack_path = "cfg.augment.esp32_stack"

    transforms: list[Any] = []
    for key, class_name in _ESP32_STACK:
        transform_cls = getattr(albumentations, class_name, None)
        if transform_cls is None:
            raise RuntimeError(
                f"albumentations {albumentations.__version__} does not provide "
                f"'{class_name}'; this module is verified against albumentations>=2.0,<3 -- "
                "fix the installed version instead of substituting a different transform"
            )
        kwargs = _to_kwargs(_node_get(stack_cfg, key, stack_path), f"{stack_path}.{key}")
        transforms.append(transform_cls(**kwargs))
    return transforms
