"""Tests for the typed config loader (trashmonkey.utils.config)."""

import dataclasses
from pathlib import Path

import pytest
import yaml

from trashmonkey.utils.config import (
    DEFAULT_CONFIG_PATH,
    Config,
    ConfigError,
    load_config,
)


@pytest.fixture(scope="module")
def config() -> Config:
    return load_config()


def _raw() -> dict[str, object]:
    with open(DEFAULT_CONFIG_PATH) as f:
        data: dict[str, object] = yaml.safe_load(f)
    return data


def _write(tmp_path: Path, data: dict[str, object]) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_default_path_resolves_without_cwd(config: Config) -> None:
    assert DEFAULT_CONFIG_PATH.is_file()
    assert isinstance(config, Config)


def test_seed_and_classes(config: Config) -> None:
    assert config.seed == 42
    assert config.classes == ("plastic", "paper", "cardboard", "metal", "glass", "organic")
    assert "rest" not in config.classes


def test_paths_are_pathlib(config: Config) -> None:
    assert config.paths.raw == Path("data/raw")
    assert isinstance(config.paths.models, Path)


def test_train_section(config: Config) -> None:
    t = config.train
    assert (t.epochs, t.optimizer, t.batch, t.imgsz) == (100, "AdamW", 16, 640)
    assert (t.lr0, t.lrf, t.momentum, t.weight_decay) == (0.001, 0.01, 0.9, 0.0005)
    assert (t.warmup_epochs, t.patience, t.close_mosaic) == (3.0, 30, 10)
    assert (t.cache, t.amp, t.deterministic, t.freeze, t.workers) == ("disk", True, True, None, 8)
    assert t.cos_lr is False
    assert t.cls_pw == 0.0  # native class weighting OFF by default (backward-compatible)


def test_augment_native_args(config: Config) -> None:
    a = config.augment
    assert (a.degrees, a.flipud, a.fliplr) == (180.0, 0.5, 0.5)
    assert (a.hsv_h, a.hsv_s, a.hsv_v) == (0.015, 0.7, 0.5)
    assert (a.translate, a.scale, a.mosaic) == (0.1, 0.5, 1.0)
    assert (a.mixup, a.cutmix, a.shear, a.perspective, a.copy_paste) == (0.0,) * 5


def test_augment_esp32_stack(config: Config) -> None:
    s = config.augment.esp32_stack
    assert s.image_compression.quality_range == (50, 85) and s.image_compression.p == 0.5
    assert s.iso_noise.p == 0.3 and s.gauss_noise.p == 0.2
    assert s.motion_blur.blur_limit == (3, 7) and s.motion_blur.p == 0.3
    assert s.defocus.radius == (1, 3) and s.defocus.p == 0.1
    assert s.planckian_jitter.p == 0.3
    assert s.downscale.scale_range == (0.4, 0.75) and s.downscale.p == 0.3
    assert s.random_brightness_contrast.p == 0.2


def test_eval_section(config: Config) -> None:
    # T6 post-census values: RealWaste is the leave-one-source-out TEST-1 source.
    assert config.eval.val_fraction == 0.15
    assert config.eval.leave_out_source == "realwaste"
    assert config.eval.test2_severities == (1, 2, 3, 4, 5)


def test_eval_clean_holdout(config: Config) -> None:
    ch = config.eval.clean_holdout
    assert ch.fraction == 0.15
    assert ch.sources == ("trashnet", "drinking-waste", "alistairking-household")


def test_eval_escalation_floors(config: Config) -> None:
    esc = config.eval.escalation
    assert (esc.overall_map50, esc.class_map50, esc.class_recall) == (0.80, 0.70, 0.70)


def test_eval_label_filter(config: Config) -> None:
    lf = config.eval.label_filter
    assert lf.min_confidence == 0.30
    assert (lf.max_box_frac, lf.min_box_frac) == (0.92, 0.005)
    assert lf.drop_methods == ("centerbox",)


def test_thresholds_section(config: Config) -> None:
    th = config.thresholds
    assert (th.conf_floor, th.tau_frame, th.min_votes, th.high_water) == (0.25, 0.40, 3, 0.60)
    assert th.sweep.tau_frame == (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
    assert th.sweep.min_votes == (2, 3, 4, 5)
    assert th.sweep.high_water == (0.5, 0.6, 0.7)


def test_config_is_frozen(config: Config) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.seed = 7  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.train.epochs = 1  # type: ignore[misc]


def test_unknown_top_level_key_raises(tmp_path: Path) -> None:
    data = _raw()
    data["banana"] = 1
    with pytest.raises(ConfigError, match=r"config: unknown key\(s\) 'banana'"):
        load_config(_write(tmp_path, data))


def test_unknown_nested_key_raises_with_section_path(tmp_path: Path) -> None:
    data = _raw()
    train = data["train"]
    assert isinstance(train, dict)
    train["lr_zero"] = 0.01
    with pytest.raises(ConfigError, match=r"config\.train: unknown key\(s\) 'lr_zero'"):
        load_config(_write(tmp_path, data))


def test_missing_key_raises(tmp_path: Path) -> None:
    data = _raw()
    thresholds = data["thresholds"]
    assert isinstance(thresholds, dict)
    del thresholds["min_votes"]
    with pytest.raises(ConfigError, match=r"config\.thresholds: missing required key\(s\): min_votes"):
        load_config(_write(tmp_path, data))


def test_wrong_type_raises(tmp_path: Path) -> None:
    data = _raw()
    train = data["train"]
    assert isinstance(train, dict)
    train["epochs"] = "a hundred"
    with pytest.raises(ConfigError, match=r"config\.train\.epochs: expected an int"):
        load_config(_write(tmp_path, data))


def test_wrong_tuple_length_raises(tmp_path: Path) -> None:
    data = _raw()
    augment = data["augment"]
    assert isinstance(augment, dict)
    augment["esp32_stack"]["image_compression"]["quality_range"] = [50, 85, 99]
    with pytest.raises(ConfigError, match=r"quality_range: expected exactly 2 items"):
        load_config(_write(tmp_path, data))


def test_null_where_not_allowed_raises(tmp_path: Path) -> None:
    data = _raw()
    train = data["train"]
    assert isinstance(train, dict)
    train["batch"] = None
    with pytest.raises(ConfigError, match=r"config\.train\.batch: expected an int, got NoneType"):
        load_config(_write(tmp_path, data))


def test_optional_fields_accept_values(tmp_path: Path) -> None:
    data = _raw()
    eval_section = data["eval"]
    assert isinstance(eval_section, dict)
    eval_section["val_fraction"] = 0.15
    eval_section["leave_out_source"] = "trashnet"
    train = data["train"]
    assert isinstance(train, dict)
    train["freeze"] = 11
    cfg = load_config(_write(tmp_path, data))
    assert cfg.eval.val_fraction == 0.15
    assert cfg.eval.leave_out_source == "trashnet"
    assert cfg.train.freeze == 11


def test_rest_class_rejected(tmp_path: Path) -> None:
    data = _raw()
    classes = data["classes"]
    assert isinstance(classes, list)
    classes.append("rest")
    with pytest.raises(ConfigError, match="'rest' is not a trained class"):
        load_config(_write(tmp_path, data))


def test_missing_file_raises() -> None:
    with pytest.raises(ConfigError, match="config file not found"):
        load_config(Path("/nonexistent/config.yaml"))
