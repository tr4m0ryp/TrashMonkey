"""Fail-fast YAML -> frozen dataclass loader.

Unknown, missing, or mistyped keys raise ``ConfigError`` naming the offending
key -- no silent fallbacks.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import UnionType
from typing import Any, TypeVar, Union, cast, get_args, get_origin, get_type_hints

import yaml

from trashmonkey.utils.config.schema import Config

_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "config.yaml"

_T = TypeVar("_T")


class ConfigError(Exception):
    """The YAML config does not match the expected schema."""


def _convert(value: object, target: Any, where: str) -> Any:
    origin = get_origin(target)
    if origin is UnionType or origin is Union:
        args = get_args(target)
        if value is None:
            if type(None) in args:
                return None
            raise ConfigError(f"{where}: null is not allowed here")
        inner = [a for a in args if a is not type(None)]
        if len(inner) != 1:
            raise ConfigError(f"{where}: unsupported union type {target!r}")
        return _convert(value, inner[0], where)
    if origin is tuple:
        args = get_args(target)
        if not isinstance(value, list):
            raise ConfigError(f"{where}: expected a list, got {type(value).__name__}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(v, args[0], f"{where}[{i}]") for i, v in enumerate(value))
        if len(value) != len(args):
            raise ConfigError(f"{where}: expected exactly {len(args)} items, got {len(value)}")
        return tuple(_convert(v, a, f"{where}[{i}]") for i, (v, a) in enumerate(zip(value, args)))
    if dataclasses.is_dataclass(target) and isinstance(target, type):
        return _build_dataclass(target, value, where)
    if target is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"{where}: expected a bool, got {type(value).__name__}")
    if target is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        raise ConfigError(f"{where}: expected an int, got {type(value).__name__}")
    if target is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        raise ConfigError(f"{where}: expected a number, got {type(value).__name__}")
    if target is str:
        if isinstance(value, str):
            return value
        raise ConfigError(f"{where}: expected a string, got {type(value).__name__}")
    if target is Path:
        if isinstance(value, str):
            return Path(value)
        raise ConfigError(f"{where}: expected a path string, got {type(value).__name__}")
    raise ConfigError(f"{where}: unsupported schema type {target!r}")


def _build_dataclass(cls: type[_T], data: object, where: str) -> _T:
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    factory: Any = cls
    hints = get_type_hints(factory)
    names = [f.name for f in dataclasses.fields(factory)]
    unknown = [k for k in data if k not in names]
    if unknown:
        keys = ", ".join(repr(k) for k in unknown)
        raise ConfigError(f"{where}: unknown key(s) {keys}; allowed keys: {', '.join(names)}")
    missing = [n for n in names if n not in data]
    if missing:
        raise ConfigError(f"{where}: missing required key(s): {', '.join(missing)}")
    kwargs = {n: _convert(data[n], hints[n], f"{where}.{n}") for n in names}
    return cast(_T, factory(**kwargs))


def _validate_classes(classes: tuple[str, ...]) -> None:
    if not classes:
        raise ConfigError("config.classes: must not be empty")
    if len(set(classes)) != len(classes):
        raise ConfigError("config.classes: duplicate class names")
    if "rest" in classes:
        raise ConfigError(
            "config.classes: 'rest' is not a trained class; "
            "it is a confidence-threshold rule in control logic"
        )


def load_config(path: Path | None = None) -> Config:
    """Load and validate the experiment config (default: configs/config.yaml)."""
    config_path = DEFAULT_CONFIG_PATH if path is None else path
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path}: top level must be a mapping")
    config = _build_dataclass(Config, raw, "config")
    _validate_classes(config.classes)
    return config
