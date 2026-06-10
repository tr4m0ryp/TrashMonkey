"""Tests for the shared ESP32 degradation module (T5 train stack + T6 severity path)."""

import sys
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from yolo_waste_sorter.utils.degrade import build_train_stack, degrade_image

# --- stub config (do NOT import the task-001 loader) -------------------------

_ESP32_STACK_PARAMS: dict[str, dict[str, Any]] = {
    "image_compression": {"quality_range": [50, 85], "p": 0.5},
    "iso_noise": {"p": 0.3},
    "gauss_noise": {"p": 0.2},
    "motion_blur": {"blur_limit": [3, 7], "p": 0.3},
    "defocus": {"radius": [1, 3], "p": 0.1},
    "planckian_jitter": {"p": 0.3},
    "downscale": {"scale_range": [0.4, 0.75], "p": 0.3},
    "random_brightness_contrast": {"p": 0.2},
}

_EXPECTED_ORDER = [
    "ImageCompression",
    "ISONoise",
    "GaussNoise",
    "MotionBlur",
    "Defocus",
    "PlanckianJitter",
    "Downscale",
    "RandomBrightnessContrast",
]


def _stub_cfg(stack: dict[str, dict[str, Any]] | None = None) -> SimpleNamespace:
    params = _ESP32_STACK_PARAMS if stack is None else stack
    return SimpleNamespace(augment=SimpleNamespace(esp32_stack=params))


def _test_image(size: int = 256) -> npt.NDArray[np.uint8]:
    """Deterministic structured image: gradient background plus shapes."""
    gradient = np.linspace(30, 220, size, dtype=np.uint8)
    img = np.stack(
        [
            np.tile(gradient, (size, 1)),
            np.tile(gradient[::-1], (size, 1)),
            np.full((size, size), 128, dtype=np.uint8),
        ],
        axis=2,
    )
    img = np.ascontiguousarray(img)
    cv2.circle(img, (size // 3, size // 3), size // 6, (250, 60, 60), -1)
    cv2.rectangle(img, (size // 2, size // 2), (size - 20, size - 20), (40, 200, 90), -1)
    cv2.putText(img, "WASTE", (10, size - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10, 10, 10), 2)
    return np.asarray(img, dtype=np.uint8)


def _psnr(a: npt.NDArray[np.uint8], b: npt.NDArray[np.uint8]) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0**2 / mse)


# --- build_train_stack (T5) ---------------------------------------------------


def test_train_stack_classes_and_order() -> None:
    stack = build_train_stack(_stub_cfg())
    assert [type(t).__name__ for t in stack] == _EXPECTED_ORDER


def test_train_stack_applies_config_params() -> None:
    by_name = {type(t).__name__: t for t in build_train_stack(_stub_cfg())}
    assert by_name["ImageCompression"].quality_range == (50, 85)
    assert by_name["ImageCompression"].p == 0.5
    assert by_name["ISONoise"].p == 0.3
    assert by_name["GaussNoise"].p == 0.2
    assert by_name["MotionBlur"].blur_limit == (3, 7)
    assert by_name["MotionBlur"].p == 0.3
    assert by_name["Defocus"].radius == (1, 3)
    assert by_name["Defocus"].p == 0.1
    assert by_name["PlanckianJitter"].p == 0.3
    assert by_name["Downscale"].scale_range == (0.4, 0.75)
    assert by_name["Downscale"].p == 0.3
    assert by_name["RandomBrightnessContrast"].p == 0.2


def test_train_stack_transforms_are_image_only() -> None:
    import albumentations

    for transform in build_train_stack(_stub_cfg()):
        assert isinstance(transform, albumentations.ImageOnlyTransform), (
            f"{type(transform).__name__} is not bbox-safe"
        )


def test_train_stack_composes_and_runs_on_cpu() -> None:
    import albumentations

    compose = albumentations.Compose(build_train_stack(_stub_cfg()), seed=42)
    out = compose(image=_test_image())["image"]
    assert out.shape == (256, 256, 3)
    assert out.dtype == np.uint8


def test_train_stack_accepts_mapping_style_cfg() -> None:
    cfg = {"augment": {"esp32_stack": _ESP32_STACK_PARAMS}}
    assert len(build_train_stack(cfg)) == len(_EXPECTED_ORDER)


def test_train_stack_missing_transform_key_fails_fast() -> None:
    stack = {k: v for k, v in _ESP32_STACK_PARAMS.items() if k != "gauss_noise"}
    with pytest.raises(KeyError, match="gauss_noise"):
        build_train_stack(_stub_cfg(stack))


def test_train_stack_missing_p_fails_fast() -> None:
    stack = dict(_ESP32_STACK_PARAMS)
    stack["iso_noise"] = {}
    with pytest.raises(KeyError, match="iso_noise.p"):
        build_train_stack(_stub_cfg(stack))


def test_train_stack_missing_albumentations_raises_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "albumentations", None)
    with pytest.raises(ImportError, match="pip install 'albumentations"):
        build_train_stack(_stub_cfg())


# --- degrade_image (T6/T9) ----------------------------------------------------


def test_degrade_image_is_byte_deterministic() -> None:
    img = _test_image()
    for severity in range(1, 6):
        first = degrade_image(img, severity, seed=42)
        second = degrade_image(img, severity, seed=42)
        assert first.tobytes() == second.tobytes(), f"severity {severity} not deterministic"


def test_degrade_image_depends_on_seed_and_severity() -> None:
    img = _test_image()
    assert degrade_image(img, 3, seed=42).tobytes() != degrade_image(img, 3, seed=43).tobytes()
    assert degrade_image(img, 2, seed=42).tobytes() != degrade_image(img, 3, seed=42).tobytes()


def test_degrade_image_psnr_decreases_with_severity() -> None:
    img = _test_image()
    psnrs = [_psnr(img, degrade_image(img, severity, seed=42)) for severity in range(1, 6)]
    assert all(earlier > later for earlier, later in zip(psnrs, psnrs[1:])), (
        f"PSNR not strictly decreasing across severities: {psnrs}"
    )
    assert psnrs[0] < float("inf"), "severity 1 must actually degrade the image"


def test_degrade_image_preserves_shape_and_dtype() -> None:
    img = _test_image(200)
    out = degrade_image(img, 5, seed=0)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert out is not img


@pytest.mark.parametrize("severity", [0, 6, -1])
def test_degrade_image_rejects_out_of_range_severity(severity: int) -> None:
    with pytest.raises(ValueError, match="severity"):
        degrade_image(_test_image(64), severity, seed=42)


def test_degrade_image_rejects_non_int_severity() -> None:
    with pytest.raises(ValueError, match="severity"):
        degrade_image(_test_image(64), 2.5, seed=42)  # type: ignore[arg-type]


def test_degrade_image_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        degrade_image(_test_image(64), 3, seed=-1)


def test_degrade_image_rejects_bad_images() -> None:
    with pytest.raises(ValueError, match="uint8"):
        degrade_image(_test_image(64).astype(np.float32), 3, seed=42)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="HxWx3"):
        degrade_image(np.zeros((64, 64), dtype=np.uint8), 3, seed=42)
